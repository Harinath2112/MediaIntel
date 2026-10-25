"""Keyword library for the MediaIntel RPA bot (Robot Framework).

Each public method is a *keyword* that rpa/mediaintel_rpa.robot calls, in the same way a
UiPath / Power Automate bot chains activities: generate the strategy report -> generate the PDF report ->
verify the files -> archive the report -> read the alert ledger -> compose the e-mail -> send it with
attachments -> mark the alerts as handled.
"""
from __future__ import annotations

import json
import shutil
import smtplib
import ssl
import sys
from datetime import datetime
from email.message import EmailMessage
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from core import anomaly_alert, config  # noqa: E402

PDF_PATH = config.OUTPUTS_DIR / "MediaIntel_Dashboard_Documentation.pdf"
SUMMARY_PATH = config.OUTPUTS_DIR / "AlertSummary.txt"
ARCHIVE_DIR = config.OUTPUTS_DIR / "archive"
RPA_DIR = config.RPA_LOG_DIR
RESULT_JSON = RPA_DIR / "dispatch_result.json"
KEEP_ARCHIVES = 30
MAX_ALERTS = max(config.env_int("MAX_ALERTS_PER_EMAIL", 10), 1)   # most alerts listed in one e-mail

ROBOT_LIBRARY_SCOPE = "GLOBAL"


class MediaIntelRPA:
    ROBOT_LIBRARY_SCOPE = "GLOBAL"

    def __init__(self):
        self._alert_ids: list[int] = []     # every alert listed in this e-mail
        self._new_ids: list[int] = []       # the unsent ones among them (marked as e-mailed after sending)
        self._pending_total = 0

    # ------------------------------------------------------------------ report generation (done BY the bot)
    def generate_strategy_report(self):
        """Builds the executive strategy report (text) from the live data."""
        from core import content_strategy
        content_strategy.generate_strategic_insights()
        return str(content_strategy.REPORT_PATH)

    def generate_pdf_report(self):
        """Builds the PDF report: dashboard charts, Whisper audio analysis, relevance, topics, anomalies, recommendations."""
        from core import generate_pdf_report
        generate_pdf_report.generate_professional_pdf()
        return str(PDF_PATH)

    # ------------------------------------------------------------------ files
    def verify_pipeline_outputs(self):
        """Fails the bot early (with a clear message) when the report or alert summary is missing."""
        if not SUMMARY_PATH.exists():
            from core import anomaly_alert
            report, total_new = anomaly_alert.alerts_for_report()
            anomaly_alert.write_summary(report, total_new)
        missing = [p.name for p in (PDF_PATH, SUMMARY_PATH) if not p.exists()]
        if missing:
            raise AssertionError(f"Pipeline output missing: {', '.join(missing)} - run the earlier stages first.")
        return str(PDF_PATH)

    def archive_report(self):
        """Stores a time-stamped copy of the PDF in /reports and keeps only the newest 30."""
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        target = ARCHIVE_DIR / f"MediaIntel_Report_{datetime.now():%Y%m%d_%H%M}.pdf"
        shutil.copy2(PDF_PATH, target)
        old = sorted(ARCHIVE_DIR.glob("MediaIntel_Report_*.pdf"))
        for f in old[:-KEEP_ARCHIVES]:
            f.unlink(missing_ok=True)
        return str(target)

    # ------------------------------------------------------------------ data
    def count_pending_alerts(self):
        """Picks the alerts for this e-mail (same layout on every run): up to MAX_ALERTS_PER_EMAIL, unsent ones first,
        topped up with recent already-sent ones. Unsent alerts above the limit wait for the next e-mail."""
        df, total_new = anomaly_alert.alerts_for_report(MAX_ALERTS)
        self._pending_total = total_new
        self._alert_ids = [int(i) for i in df["id"].tolist()]
        self._new_ids = [int(i) for i in df.loc[df["is_new"], "id"].tolist()]
        return len(self._alert_ids)

    # ------------------------------------------------------------------ e-mail
    def compose_email(self, alert_count):
        """Builds subject + plain/HTML body from the live database. Returns a dict."""
        alert_count = int(alert_count)
        ids = self._alert_ids or [-1]
        alerts = config.read_sql(f"""
            SELECT a.metric, a.velocity, a.baseline, a.ratio, a.severity, a.emailed, v.title, c.channel_name
            FROM anomaly_alerts a JOIN videos v ON v.video_id = a.video_id
            JOIN channels c ON c.channel_id = a.channel_id
            WHERE a.id IN ({", ".join(str(int(i)) for i in ids)}) ORDER BY a.ratio DESC""")
        kpi = config.read_sql("""
            SELECT (SELECT COUNT(*) FROM channels WHERE is_tracked = TRUE) AS channels,
                   (SELECT COUNT(*) FROM videos v JOIN channels c ON c.channel_id = v.channel_id WHERE c.is_tracked = TRUE) AS videos,
                   (SELECT COUNT(*) FROM sentiment_insights s JOIN comments m ON m.comment_id = s.comment_id
                      JOIN videos v ON v.video_id = m.video_id JOIN channels c ON c.channel_id = v.channel_id
                      WHERE c.is_tracked = TRUE) AS comments,
                   (SELECT AVG(r.relevance_index) FROM video_relevance r JOIN videos v ON v.video_id = r.video_id
                      JOIN channels c ON c.channel_id = v.channel_id WHERE c.is_tracked = TRUE) AS relevance""").iloc[0]
        topics = config.read_sql("SELECT topic, score, evidence FROM topic_suggestions "
                                 "WHERE scope = 'ALL' AND rank_no <= 5 ORDER BY rank_no")
        unit = {"views": "views/hr", "likes": "likes/hr", "comments": "comments/hr"}
        rel_val = round(float(kpi.relevance), 1) if pd.notna(kpi.relevance) else 0.0

        if alert_count:
            top = alerts.iloc[0]
            level = str(top.severity or "MEDIUM").upper()
            subject = (f"[{level} ALERT] MediaIntel: {alert_count} engagement anomal"
                       f"{'y' if alert_count == 1 else 'ies'} - '{top.channel_name}' is trending")
            intro = "Automated anomaly alert from the MediaIntel platform."
        else:
            subject = f"[MediaIntel] Competitive intelligence report - {datetime.now():%d %b %Y}"
            intro = "No engagement anomalies recorded recently. Your latest competitive intelligence report is attached."

        text_lines = [intro, ""]
        html = [f"<h2 style='color:#1a365d'>{escape(intro)}</h2>"]
        if len(alerts):
            text_lines.append("ANOMALIES")
            html.append("<h3>Anomaly signals</h3><table border='1' cellpadding='6' cellspacing='0' "
                        "style='border-collapse:collapse;font-size:13px'><tr style='background:#2b6cb0;color:#fff'>"
                        "<th>Status</th><th>Severity</th><th>Channel</th><th>Video</th><th>Metric</th><th>Rate</th><th>Baseline</th><th>x</th></tr>")
            for a in alerts.itertuples(index=False):
                text_lines.append(f" - [{'NEW' if not a.emailed else 'EARLIER'}] [{a.severity}] [{a.metric}] {a.channel_name}: {a.title} -> {a.velocity:.1f} {unit[a.metric]} "
                                  f"(baseline {a.baseline:.1f}, {a.ratio:.1f}x)")
                html.append(f"<tr><td>{'NEW' if not a.emailed else 'Earlier'}</td><td><b>{a.severity}</b></td><td>{escape(a.channel_name)}</td><td>{escape(a.title)}</td><td>{a.metric}</td>"
                            f"<td>{a.velocity:.1f} {unit[a.metric]}</td><td>{a.baseline:.1f}</td><td><b>{a.ratio:.1f}x</b></td></tr>")
            html.append("</table>")
            text_lines.append("")
            if not self._new_ids:
                note = "No new anomalies since the last e-mail - the strongest recent alerts are repeated above for reference."
                text_lines += [note, ""]
                html.append(f"<p style='color:#475569'>{escape(note)}</p>")
            extra = self._pending_total - len(self._new_ids)
            if extra > 0:
                note = (f"{extra} more alert(s) are waiting and will be sent in the next e-mail "
                        f"(see the Alerts tab of the dashboard).")
                text_lines += [note, ""]
                html.append(f"<p style='color:#92400e'>{escape(note)}</p>")
        text_lines.append(f"Tracked competitors: {int(kpi.channels)} | Videos: {int(kpi.videos)} | "
                          f"Comments analysed: {int(kpi.comments)} | Avg relevance index: {rel_val}")
        html.append(f"<p><b>Tracked competitors:</b> {int(kpi.channels)} &nbsp; <b>Videos:</b> {int(kpi.videos)} &nbsp; "
                    f"<b>Comments analysed:</b> {int(kpi.comments)} &nbsp; <b>Avg relevance index:</b> {rel_val}</p>")
        if len(topics):
            text_lines += ["", "TOP TOPICS FOR YOUR NEXT VIDEO"]
            html.append("<h3>Top topics for your next video</h3><ol>")
            for t in topics.itertuples(index=False):
                text_lines.append(f" {t.topic} (score {t.score:.0f}) - {t.evidence}")
                html.append(f"<li><b>{escape(t.topic)}</b> (score {t.score:.0f})<br><span style='color:#475569'>"
                            f"{escape(t.evidence)}</span></li>")
            html.append("</ol>")
        text_lines += ["", "The full PDF report and alert summary are attached."]
        html.append("<p style='color:#64748b'>The full PDF report and alert summary are attached.<br>"
                    "Sent automatically by the MediaIntel RPA bot.</p>")
        return {"subject": subject, "text": "\n".join(text_lines), "html": "\n".join(html), "alerts": alert_count,
                "new_alerts": len(self._new_ids)}

    def send_email_with_attachments(self, mail):
        """Sends the message via SMTP (or only saves it when RPA_DRY_RUN=1). Returns 'SENT' / 'DRY_RUN' / 'SKIPPED'."""
        RPA_DIR.mkdir(parents=True, exist_ok=True)
        (RPA_DIR / "outbox").mkdir(exist_ok=True)

        if mail.get("new_alerts", mail["alerts"]) == 0 and config.env("EMAIL_MODE", "always").lower() == "alerts_only":
            self._result(mail, "SKIPPED", note="EMAIL_MODE=alerts_only and no anomalies")
            return "SKIPPED"

        mail_cfg = config.mail_settings()
        host, port, sender = mail_cfg["host"], mail_cfg["port"], mail_cfg["sender"]
        password, recipients = mail_cfg["password"], mail_cfg["recipients"]
        dry = config.env_bool("RPA_DRY_RUN", False)
        if not dry and not (sender and recipients and password):
            dry = True

        msg = EmailMessage()
        msg["From"] = sender or "mediaintel@localhost"
        msg["To"] = ", ".join(recipients) or "demo@localhost"
        msg["Subject"] = mail["subject"]
        msg.set_content(mail["text"])
        msg.add_alternative(mail["html"], subtype="html")
        for path, mime in ((PDF_PATH, ("application", "pdf")), (SUMMARY_PATH, ("text", "plain"))):
            msg.add_attachment(path.read_bytes(), maintype=mime[0], subtype=mime[1], filename=path.name)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (RPA_DIR / "outbox" / f"mail_{stamp}.eml").write_bytes(bytes(msg))   # audit trail of every e-mail

        if dry:
            self._result(mail, "DRY_RUN", note="RPA_DRY_RUN=1 - saved to data/rpa_logs/outbox only")
            return "DRY_RUN"

        security = (config.env("SMTP_SECURITY") or ("ssl" if port == 465 else "starttls")).lower()
        try:
            if security == "ssl":
                server = smtplib.SMTP_SSL(host, port, timeout=30, context=ssl.create_default_context())
            else:
                server = smtplib.SMTP(host, port, timeout=30)
                if security == "starttls":
                    server.starttls(context=ssl.create_default_context())
            with server:
                if password:
                    server.login(sender, password)
                server.send_message(msg, to_addrs=recipients)
        except Exception as exc:
            self._result(mail, "FAILED", note=str(exc)[:200])
            raise AssertionError(f"SMTP delivery failed: {exc}") from exc
        self._result(mail, "SENT", note=f"to {', '.join(recipients)}")
        return "SENT"

    def mark_alerts_emailed(self, status):
        """Closes the loop: alerts that reached the inbox are never e-mailed again."""
        if status != "SENT" or not self._new_ids:
            return 0
        from sqlalchemy import bindparam, text
        with config.get_engine().begin() as conn:
            conn.execute(text("UPDATE anomaly_alerts SET emailed = 1 WHERE id IN :ids")
                         .bindparams(bindparam("ids", expanding=True)), {"ids": self._new_ids})
        return len(self._new_ids)

    # ------------------------------------------------------------------ result file for the orchestrator
    def _result(self, mail, status, note=""):
        RPA_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_JSON.write_text(json.dumps({
            "status": status, "alerts": mail["alerts"], "subject": mail["subject"], "note": note,
            "time": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
