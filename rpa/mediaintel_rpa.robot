*** Settings ***
Documentation     MediaIntel RPA bot (Robot Framework, RPA mode).
...               Does the work an analyst would otherwise do by hand after every analysis run:
...               generate the strategy report and the PDF report, check and archive the files,
...               collect the anomaly alerts, write the e-mail, attach the PDF + alert summary,
...               send it, and mark the alerts as handled.
...               Runs on any laptop: `pip install robotframework` - no licence, no robot installer.
Library           MediaIntelRPA.py

*** Tasks ***
Generate Reports And Dispatch Alerts
    ${strategy}=      Generate Strategy Report
    Log               Strategy report written to ${strategy}
    ${pdf}=           Generate PDF Report
    Log               PDF report written to ${pdf}
    ${verified}=      Verify Pipeline Outputs
    ${archived}=      Archive Report
    Log               Report archived at ${archived}
    ${alerts}=        Count Pending Alerts
    Log               ${alerts} anomaly alert(s) waiting to be e-mailed
    ${mail}=          Compose Email    ${alerts}
    Log               Subject: ${mail}[subject]
    ${status}=        Send Email With Attachments    ${mail}
    ${closed}=        Mark Alerts Emailed    ${status}
    Log               E-mail status: ${status}; ${closed} alert(s) closed
