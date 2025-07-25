# Climate-Adaptation Insurance Report Agent

*Autonomously discovers, emails for approval, and stores reports on climate-adaptation insurance.*

## First-time setup
1. Add GitHub Secrets:
   * GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID
   * SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD
   * REVIEWER_EMAIL
2. Click **Actions → Crawl → Run workflow**.
3. Check your inbox → approve / reject each candidate.  
   Reject = `REJECT sha256`, Hard negative = `NEVER sha256`.

## Weekly cycle
* Crawl runs every 7 days → sends email
* Your clicks write `data/labels/YYYY-MM-DD.csv`
* `Train` workflow retrains classifier automatically
