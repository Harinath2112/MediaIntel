"""Stage 4 - Content relevance + "what should the next video be about?" topic mining.

Uses the Whisper transcripts and the viewers' comments together.

A) RELEVANCE (per video, stored in `video_relevance`)
   Multilingual embeddings compare what the video SAYS (transcript chunks) with what viewers WRITE.
     comment_alignment : mean similarity of the video's comments to the closest part of the transcript
     on_topic_share    : share of comments that clearly talk about the video's actual content
     title_alignment   : does the transcript deliver what the title promises
     relevance_index   : 100 * (0.5*comment_alignment + 0.3*on_topic_share + 0.2*title_alignment)
   It is a similarity measure (how closely discussion follows content), not a quality verdict.

B) NEXT-VIDEO TOPICS (stored in `topic_suggestions`, for all channels and for each channel)
   Candidate topics are mined from comments, then ranked by
     demand (how many viewers mention it, liked comments count more)           40 %
     explicit requests ("please make a video on ...", "... ka video banao")    30 %
     content gap (how little the existing transcripts already cover it)        20 %
     audience sentiment around the topic                                       10 %
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402
from sqlalchemy import text  # noqa: E402

# ------------------------------------------------------------------ tunables
SIM_LOW, SIM_HIGH = 0.10, 0.55      # cosine range mapped to 0..1 for the relevance index
ON_TOPIC_SIM = 0.35                 # comment is "on topic" when this similar to some part of the video
COVER_SIM = 0.45                    # a topic counts as covered when a transcript chunk is this similar
CHUNK_WORDS, CHUNK_STRIDE, MAX_CHUNKS = 60, 40, 40
MAX_COMMENTS_PER_VIDEO = config.env_int("RELEVANCE_COMMENTS_PER_VIDEO", 300)   # comments embedded per video (0 = all)
TOP_K = config.env_int("TOPICS_TOP_K", 10)
MIN_TRANSCRIPT_WORDS = 20

# ------------------------------------------------------------------ language helpers
STOP = set("""
a about above after again all also am an and any are as at be because been before being below between both but by can could did do does
doing down during each few for from further had has have having he her here hers him his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours out over own same she should so some such than that the their theirs
them then there these they this those through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves
also get got getting like likes liked make made making want wanted wants need needs needed please pls plz kindly sir sirji ji mam madam
maam bhai bhaiya bhaiyya bro brother didi anna akka guru guruji dear friend friends guys guy thanks thank thankyou thx tysm love loved
loving best good great nice awesome amazing super superb excellent wonderful fantastic perfect better well fine cool bad worst
really truly very much many lot lots one two three first second last next new old more less
hi hello hey wow ok okay yes yeah yep hmm lol haha omg
ka ki ke ko se me mein mai main hai hain ho hoga hogi hota hoti tha thi the thi kya kyu kyun kaise kaisa kab kaun kis jo ye wo woh yeh
aur ya bhi hi toh to par pe pr per lekin magar abhi ab phir bas sirf bahut bohot bhut kuch koi sab sabhi apne apna apni aap aapka aapki
aapke aapko hum humko hume mujhe mera meri mere tum tumhare tera teri tere unka unki unke iska iski iske liye lie wala wali wale karo
kare kar kara karna karne karke karta karti kiya kiye diya diye dena dene lena lene aya aaya aayega aaye aaja aao chahiye chahie chaiye
hona hone raha rahi rahe rha rhe rhi gaya gayi gaye lagta lagti sakta sakte sakti sir's
""".split())
# Words that carry no topic on their own and always break a phrase
GENERIC = set("""
video videos vid channel playlist series course courses tutorial tutorials class classes lecture lectures upload uploads uploaded
uploading post posted watch watched watching subscribe subscribed subscriber subscribers like comment comments share link links
day days time times year years month months week weeks hour hours minute minutes sec second seconds today tomorrow yesterday
everyone everybody anyone someone thing things stuff way ways part parts full complete whole sorry cover covers covered covering
banao banado banaiye banaiyega bana scratch beginner beginners basic basics advanced start starting started begin
waiting wait waited soon request requests requesting requested again already anyway everything nothing something
back bring brought know knew known knowing come came coming comes go going gone goes give gave given giving take took taken taking
tell told telling say said saying says see saw seen look looks looking think thought thinking try tried trying use used using
work works worked working help helps helped helping keep kept let put find found finding ask asked asking show showed shown showing
call called feel felt feeling happen happened happens mean means meant read reading write wrote written turn turned stop stopped
run running ran play played give person people student students guy guys man men girl boy kid kids everyone sure maybe actually
literally definitely probably finally exactly simply honestly truly hope hoping wish wishing lots pls plz thing anything someone
sometimes always never ever still even yet away around enough able left right wrong real true hard easy long short big small little
high low top bottom home house world life india indian telugu hindi english tamil kannada malayalam bengali marathi
nhi nahi nahin mil milega milta join joined joining extend extended helpful reply replies replied website site page pages
youtube youtuber youtubers yt instagram insta telegram whatsapp facebook twitter linkedin reel reels shorts short
happy sad congratulations congratulation congrats birthday anniversary wishes wish blessed blessing diwali christmas
free giveaway winner winners offer offers discount coupon code sponsor sponsored sponsorship ad ads advertisement
don doesn didn isn wasn aren weren couldn wouldn shouldn won hasn haven hadn ll ve re ain cant dont
""".split())

# Too vague to be a topic on their own (fine inside a phrase such as "system design" or "data science")
VAGUE_SINGLE = set("""
system systems design architecture operating dynamic binary code coding app apps notes engineering engineer science analysis
data programming network networks security branch college exam exams class classes lecture lecture course courses video videos
""".split())
# Words allowed INSIDE a phrase ("machine learning", "system design interview") but never a topic by themselves
WEAK = set("""
learning learn learned teach teaching taught explain explained explaining explanation concept concepts topic topics example
examples project projects content roadmap interview interviews questions question answer answers
""".split())
SHORT_OK = {"ai", "ml", "os", "js", "ui", "ux", "go", "db", "dl", "hr", "cs", "it", "pm", "qa", "c", "r", "dp", "oop", "sql"}
ACRONYMS = {"dsa", "sql", "ai", "ml", "os", "dbms", "oop", "oops", "api", "aws", "gcp", "llm", "mcp", "rag", "nlp", "html", "css", "js",
            "ui", "ux", "gate", "cpp", "dp", "cn", "ds", "iot", "gui", "ide", "jwt", "rest", "ci", "cd", "devops", "mern", "mean", "jee"}

_SPLIT_CLAUSE = re.compile(r"[,;:!?\n()|/\[\]\"“”]+|\.(?:\s|$)")
_TOKEN = re.compile(r"[a-z][a-z0-9+#.]*")

_REQUEST_PATTERNS = [re.compile(p, re.I) for p in [
    r"\b(?:please|pls|plz|kindly)\b.{0,25}\b(?:make|create|upload|cover|teach|start|bring|do|add|post)\b",
    r"\b(?:make|create|upload|cover|teach|start|bring|do|post)\b.{0,20}\b(?:video|videos|series|playlist|course|tutorial|lecture|lectures)\b",
    r"\b(?:video|series|playlist|course|tutorial|lecture)\b.{0,12}\b(?:on|about|for|regarding)\b",
    r"\b(?:waiting for|we want|i want|need a|request(?:ing)? (?:a )?(?:video|series))\b",
    r"\b(?:banao|bana do|banaiye|banaiyega|laao|lao|chahiye|chahie|upload karo|kab aayega|kab aaega|kab ayega)\b",
]]


def is_request(comment: str) -> bool:
    return any(p.search(comment) for p in _REQUEST_PATTERNS)


def _tokens(chunk: str) -> list[str]:
    return [t.rstrip(".") for t in _TOKEN.findall(chunk.lower())]


def _usable(tok: str, extra_stop: set[str]) -> bool:
    if tok in STOP or tok in GENERIC or tok in extra_stop:
        return False
    return len(tok) >= 3 or tok in SHORT_OK


def ngrams(comment: str, extra_stop: set[str], max_n: int = 3) -> set[str]:
    """All 1-3 word phrases made only of content words (no stop/generic words inside)."""
    text_ = re.sub(r"https?://\S+|@\w+", " ", str(comment))
    found: set[str] = set()
    for clause in _SPLIT_CLAUSE.split(text_):
        run: list[str] = []
        for tok in _tokens(clause) + [None]:
            if tok is not None and _usable(tok, extra_stop):
                run.append(tok)
                continue
            for i in range(len(run)):
                for n in range(1, max_n + 1):
                    if i + n <= len(run):
                        gram = run[i:i + n]
                        if gram[0] in WEAK and n > 1:      # "learning python" -> only "python"
                            continue
                        if all(t in WEAK for t in gram):   # never a topic on its own
                            continue
                        found.add(" ".join(gram))
            run = []
    return found


# Words that are always a legitimate subject for a tutorial/education channel (extends the title vocabulary)
TECH = set("""
python java javascript typescript kotlin swift golang rust php ruby scala react angular vue node nodejs express django flask fastapi spring
springboot html css sass tailwind bootstrap sql mysql postgres postgresql mongodb redis firebase nosql dbms docker kubernetes linux unix bash
git github gitlab devops cloud aws azure gcp terraform jenkins excel powerbi tableau pandas numpy matplotlib scikit sklearn tensorflow pytorch
keras opencv nlp llm llms gpt chatgpt gemini claude openai langchain rag mcp agent agents agentic genai ai ml dl ds dsa algorithm algorithms
array arrays string strings linkedlist stack queue tree trees graph graphs heap hashing hashmap recursion backtracking greedy dynamic
programming sorting searching binary leetcode codeforces codechef hackerrank placement placements resume cv internship internships
interview interviews hr aptitude reasoning verbal coding code compiler operating oops oop oods lld hld system design architecture
microservices api apis rest graphql websocket security cybersecurity hacking hacker ethical pentest network networking networks cn os
blockchain web3 solidity android ios flutter dart reactnative app apps frontend backend fullstack mern mean data science analytics
analysis analyst engineer engineering software developer testing selenium automation rpa uipath iot embedded arduino raspberry vlsi
gate jee neet upsc cat ssc bank banking tcs infosys wipro accenture cognizant amazon google microsoft meta faang mnc startup
physics chemistry maths mathematics math biology calculus algebra statistics probability electronics circuits thermodynamics mechanics
job jobs career salary roadmap sheet sheets notes pdf cheatsheet certification certificate course bootcamp fresher freshers
""".split())


def _stem(tok: str) -> str:
    """Crude singular form so 'sheets' ~ 'sheet', 'jobs' ~ 'job', 'queries' ~ 'query'."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def domain_terms(titles) -> set[str]:
    """Vocabulary a topic must connect to: words used in the competitors' own video titles + known tech/education terms.
    This is what keeps random comment words ('back', 'bring', 'know') out of the topic list."""
    vocab = {_stem(t) for t in TECH} | {_stem(t) for t in ACRONYMS}
    for title in titles:
        for tok in _tokens(str(title or "")):
            if tok not in STOP and tok not in GENERIC and len(tok) >= 3:
                vocab.add(_stem(tok))
    return vocab


def _anchored(phrase: str, vocab: set[str]) -> bool:
    return any(_stem(t) in vocab for t in phrase.split())


def pretty(phrase: str) -> str:
    return " ".join(w.upper() if w in ACRONYMS else w.capitalize() for w in phrase.split())


# ------------------------------------------------------------------ transcript chunks
def chunk_text(body: str) -> list[str]:
    words = str(body).split()
    if len(words) <= CHUNK_WORDS:
        return [" ".join(words)] if words else []
    chunks = [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words) - CHUNK_WORDS // 2, CHUNK_STRIDE)]
    if len(chunks) > MAX_CHUNKS:  # sample evenly across the whole transcript
        idx = np.linspace(0, len(chunks) - 1, MAX_CHUNKS).astype(int)
        chunks = [chunks[i] for i in idx]
    return chunks


def video_text(row) -> tuple[str, str]:
    """(text, source) used to represent what a video says."""
    body = row.get("transcript_text")
    src = row.get("tsource")
    if isinstance(body, str) and len(body.split()) >= MIN_TRANSCRIPT_WORDS and src in ("whisper", "captions"):
        return body, src
    fallback = f"{row.get('title') or ''}. {row.get('description') or ''}"
    return fallback.strip(), "title_only"


def _norm_sim(s):
    return np.clip((np.asarray(s, dtype=float) - SIM_LOW) / (SIM_HIGH - SIM_LOW), 0, 1)


# ------------------------------------------------------------------ data loading
def load_data():
    videos = config.read_sql("""
        SELECT v.video_id, v.channel_id, v.title, v.description, c.channel_name,
               t.source AS tsource, t.transcript_text, t.updated_at AS transcript_updated_at
        FROM videos v JOIN channels c ON c.channel_id = v.channel_id
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        WHERE c.is_tracked = 1""")
    comments = config.read_sql("""
        SELECT c.comment_id, c.video_id, c.comment_text, COALESCE(c.like_count, 0) AS like_count,
               v.channel_id, s.polarity, s.aspect_category
        FROM comments c JOIN videos v ON v.video_id = c.video_id
        JOIN channels ch ON ch.channel_id = v.channel_id
        LEFT JOIN sentiment_insights s ON s.comment_id = c.comment_id
        WHERE ch.is_tracked = 1""")
    requests_ = config.read_sql("SELECT DISTINCT comment_id FROM aspect_sentiments WHERE aspect = 'Feature Request'")
    existing = config.read_sql("SELECT video_id, n_comments, text_source, computed_at FROM video_relevance")
    return videos, comments, set(requests_["comment_id"]), existing


# ------------------------------------------------------------------ A) relevance
def compute_relevance(videos, comments, existing, embedder):
    """Returns (relevance rows to upsert, {video_id: chunk embedding matrix} for ALL videos)."""
    by_video = {vid: g for vid, g in comments.groupby("video_id")}
    ex = existing.set_index("video_id") if not existing.empty else pd.DataFrame()
    rows, chunk_emb = [], {}

    # transcript chunk embeddings for every video (cheap; needed for topic coverage too)
    texts, src_map = {}, {}
    for _, v in videos.iterrows():
        body, src = video_text(v)
        texts[v.video_id], src_map[v.video_id] = body, src
    all_chunks, owners = [], []
    for vid, body in texts.items():
        for ch in chunk_text(body):
            all_chunks.append(ch)
            owners.append(vid)
    if all_chunks:
        emb = embedder.encode(all_chunks)
        for vid in set(owners):
            chunk_emb[vid] = emb[[i for i, o in enumerate(owners) if o == vid]]

    for _, v in videos.iterrows():
        vid = v.video_id
        g = by_video.get(vid)
        if g is None or g.empty or vid not in chunk_emb:
            continue
        g = g.sort_values("like_count", ascending=False)
        if MAX_COMMENTS_PER_VIDEO > 0:
            g = g.head(MAX_COMMENTS_PER_VIDEO)
        n = len(g)
        stale = True
        if not ex.empty and vid in ex.index:
            prev = ex.loc[vid]
            t_upd = v.get("transcript_updated_at")
            stale = (int(prev["n_comments"]) != n or prev["text_source"] != src_map[vid]
                     or (pd.notna(t_upd) and pd.notna(prev["computed_at"]) and t_upd > prev["computed_at"]))
        if not stale:
            continue

        title_vec = embedder.encode([str(v.title)])[0]
        ce = chunk_emb[vid]
        anchors = np.vstack([ce, title_vec[None, :]])          # comments may also react to the title itself
        com_vecs = embedder.encode(g["comment_text"].astype(str).tolist())
        best = (com_vecs @ anchors.T).max(axis=1)              # closest part of the video, per comment
        centroid = ce.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        title_align = float(_norm_sim(title_vec @ centroid)) if src_map[vid] != "title_only" else None

        comment_alignment = float(_norm_sim(best).mean())
        on_topic = float((best >= ON_TOPIC_SIM).mean())
        ta = title_align if title_align is not None else comment_alignment   # no transcript: neutral weight
        index = 100 * (0.5 * comment_alignment + 0.3 * on_topic + 0.2 * ta)
        rows.append({"video_id": vid, "relevance_index": round(index, 2),
                     "comment_alignment": round(comment_alignment, 4), "on_topic_share": round(on_topic, 4),
                     "title_alignment": None if title_align is None else round(title_align, 4),
                     "n_comments": n, "text_source": src_map[vid]})
    return rows, chunk_emb


# ------------------------------------------------------------------ B) topics
def mine_topics(comments: pd.DataFrame, request_ids: set, videos_scope: pd.DataFrame, chunk_emb: dict,
                embedder, extra_stop: set[str], vocab: set[str] | None = None) -> list[dict]:
    n_comments = len(comments)
    if n_comments == 0:
        return []
    min_mentions = max(3, math.ceil(0.003 * n_comments))

    mentions, weighted, requests_, pol_sum, pol_n = Counter(), Counter(), Counter(), Counter(), Counter()
    for r in comments.itertuples(index=False):
        if r.aspect_category in ("Technical Issue", "Pricing / Cost"):
            continue   # complaints about audio/price are not topics to cover
        grams = ngrams(r.comment_text, extra_stop)
        if not grams:
            continue
        asked = (r.comment_id in request_ids) or is_request(str(r.comment_text))
        w = 1.0 + math.log1p(max(int(r.like_count or 0), 0))
        pol = None if pd.isna(r.polarity) else float(r.polarity)
        for g in grams:
            mentions[g] += 1
            weighted[g] += w
            if asked:
                requests_[g] += 1
            if pol is not None:
                pol_sum[g] += pol
                pol_n[g] += 1

    cands = {g for g, c in mentions.items() if c >= min_mentions or requests_[g] >= 2}
    cands = {g for g in cands if " " in g or _stem(g) not in {_stem(v) for v in VAGUE_SINGLE}}
    if vocab:   # keep only meaningful subjects: the phrase must contain a word from the video titles / tech vocabulary
        cands = {g for g in cands if _anchored(g, vocab)}
    # drop a shorter phrase when a longer phrase containing it is about as frequent ("system" vs "system design")
    for g in sorted(cands, key=lambda x: -len(x.split())):
        for other in list(cands):
            if other != g and len(other.split()) < len(g.split()) and re.search(rf"\b{re.escape(other)}\b", g) \
                    and mentions[other] <= mentions[g] * 1.25:
                cands.discard(other)
    if not cands:
        return []
    cands = sorted(cands)

    # merge near-duplicate phrases (e.g. "dsa" ~ "data structures") by embedding similarity
    cvec = embedder.encode(cands)
    keep: list[int] = []
    for i in sorted(range(len(cands)), key=lambda k: -weighted[cands[k]]):
        if all(float(cvec[i] @ cvec[j]) < 0.88 for j in keep):
            keep.append(i)
    cands = [cands[i] for i in keep]
    cvec = cvec[keep]

    # coverage: how much of the existing video content already talks about the topic
    scope_ids = videos_scope["video_id"].tolist()
    body_lc = {r.video_id: video_text(r._asdict())[0].lower() for r in videos_scope.itertuples(index=False)}
    covered = []
    for k, phrase in enumerate(cands):
        pat = re.compile(rf"\b{re.escape(phrase)}\b")
        hit = 0
        for vid in scope_ids:
            lexical = bool(pat.search(body_lc.get(vid, "")))
            semantic = vid in chunk_emb and float((chunk_emb[vid] @ cvec[k]).max()) >= COVER_SIM
            hit += 1 if (lexical or semantic) else 0
        covered.append(hit / max(len(scope_ids), 1))

    max_w = max(math.log1p(weighted[c]) for c in cands) or 1.0
    max_r = max(math.log1p(requests_[c]) for c in cands) or 1.0
    out = []
    for phrase, cov in zip(cands, covered):
        avg_pol = pol_sum[phrase] / pol_n[phrase] if pol_n[phrase] else 0.0
        demand = math.log1p(weighted[phrase]) / max_w
        req = math.log1p(requests_[phrase]) / max_r if requests_[phrase] else 0.0
        score = 100 * (0.40 * demand + 0.30 * req + 0.20 * (1 - cov) + 0.10 * ((avg_pol + 1) / 2))
        covered_n = round(cov * len(scope_ids))
        pos_pct = round(100 * (avg_pol + 1) / 2)
        evidence = (f"Mentioned in {mentions[phrase]} comments"
                    + (f" ({requests_[phrase]} explicit request{'s' if requests_[phrase] != 1 else ''})" if requests_[phrase] else "")
                    + f"; already covered in {covered_n} of {len(scope_ids)} videos; audience mood {pos_pct}% positive.")
        out.append({"topic": pretty(phrase), "score": round(score, 2), "mentions": int(mentions[phrase]),
                    "request_count": int(requests_[phrase]), "avg_polarity": round(avg_pol, 3),
                    "transcript_coverage": round(cov, 3), "evidence": evidence[:490]})
    out.sort(key=lambda r: (-r["score"], -r["mentions"]))
    return out[:TOP_K]


# ------------------------------------------------------------------ orchestration
def run() -> None:
    from core import nlp_models
    print("Loading transcripts, comments and sentiment...")
    videos, comments, request_ids, existing = load_data()
    if videos.empty or comments.empty:
        print("No videos/comments yet - nothing to analyse.")
        return
    embedder = nlp_models.get_embedder()

    print("Measuring transcript <-> comment relevance...")
    rows, chunk_emb = compute_relevance(videos, comments, existing, embedder)
    if rows:
        stamp = config.utcnow()
        cols = ["video_id", "relevance_index", "comment_alignment", "on_topic_share", "title_alignment",
                "n_comments", "text_source", "computed_at"]
        with config.get_engine().begin() as conn:
            conn.execute(config.upsert_sql("video_relevance", cols, ["video_id"]), [dict(r, computed_at=stamp) for r in rows])
    print(f"  relevance computed for {len(rows)} video(s) (others unchanged)")

    print("Mining topics for the next video...")
    channel_tokens = {t for name in videos["channel_name"].unique() for t in _tokens(name)}
    extra_stop = channel_tokens | {t for name in videos["channel_name"].unique() for t in re.findall(r"[a-z]+", name.lower())}
    # optional personal block-list in .env, e.g. creator names that are not topics: TOPIC_EXTRA_STOPWORDS=striver,alakh
    extra_stop |= {w.strip().lower() for w in (config.env("TOPIC_EXTRA_STOPWORDS", "") or "").split(",") if w.strip()}

    vocab = domain_terms(videos["title"].tolist())
    scopes = [("ALL", comments, videos)] + [
        (cid, comments[comments["channel_id"] == cid], videos[videos["channel_id"] == cid])
        for cid in videos["channel_id"].unique()]
    all_rows = []
    for scope, c_scope, v_scope in scopes:
        topics = mine_topics(c_scope, request_ids, v_scope, chunk_emb, embedder, extra_stop, vocab)
        for rank, t in enumerate(topics, 1):
            all_rows.append(dict(t, scope=scope, rank_no=rank))
    with config.get_engine().begin() as conn:
        conn.execute(text("DELETE FROM topic_suggestions"))
        if all_rows:
            conn.execute(text("""
                INSERT INTO topic_suggestions (scope, rank_no, topic, score, mentions, request_count,
                                               avg_polarity, transcript_coverage, evidence)
                VALUES (:scope, :rank_no, :topic, :score, :mentions, :request_count, :avg_polarity,
                        :transcript_coverage, :evidence)"""), all_rows)
    top = [r for r in all_rows if r["scope"] == "ALL"][:5]
    print(f"  {len(all_rows)} topic suggestions stored. Top picks across all channels:")
    for r in top:
        print(f"    {r['rank_no']}. {r['topic']}  (score {r['score']:.0f}, {r['mentions']} mentions)")


if __name__ == "__main__":
    run()
