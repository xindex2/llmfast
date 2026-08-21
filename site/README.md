# llmfa.st — public website

Three static pages. No build step, no framework, no JavaScript beyond a line
that sets the year in the footer.

```
index.html    home page, with the hand-drawn explainer diagram
terms.html    Terms of Use          -> https://llmfa.st/terms
privacy.html  Privacy Policy        -> https://llmfa.st/privacy
style.css     shared styles
```

## Before you publish

The legal pages are complete and contain no placeholders. They are written for
llmfa.st, state zero data retention, and say there are no cookies or analytics —
all three of which describe what the software in this repo actually does.

Two things to keep true:

- **Retention.** Terms §9 and Privacy §3 both state that prompts and completions
  are never written to disk. That is accurate: the gateway records model, token
  counts, latency and status, and nothing else. If you ever add prompt logging,
  change both pages and set `compliance.zdr: false` on every model in
  `config/config.yaml`. Claiming zero retention while keeping content is the kind
  of thing that surfaces in an enterprise audit.
- **No analytics.** Privacy §6 says this site sets no cookies and runs no
  tracking. It does not today. If you add any, that paragraph has to change and
  you will probably need a consent banner.

The pages carry no jurisdiction or company registration number, because those
are facts about a legal filing rather than something a template can supply.
Section 25 is written to work without naming a country. Once you register a
company, have a lawyer read both pages and add the specifics — liability caps
and warranty disclaimers in particular are unenforceable as drafted in parts of
the EU and UK.

## Publishing

The site is static, so anything will serve it. Free options:

```bash
# Cloudflare Pages
npx wrangler pages deploy site --project-name llmfast

# Netlify
npx netlify deploy --dir=site --prod
```

Or copy it behind nginx on any box:

```nginx
server {
    listen 443 ssl;
    server_name llmfa.st www.llmfa.st;
    root /var/www/llmfast;
    index index.html;
    # /terms and /privacy without the .html extension, which is what the
    # OpenRouter application form should point at.
    location / { try_files $uri $uri.html $uri/ =404; }
}
```

Keep the marketing site on `llmfa.st` and the API on `api.llmfa.st`. They are
different workloads: the API must not be sharing a process with something that
search engine crawlers hit.

## Keeping it honest

The home page deliberately carries **no prices and no model version numbers**.
Both drift the moment you change `config/config.yaml`, and a marketing page that
disagrees with your own API is worse than one that says less. The model cards
name families only, and everything specific points at
`api.llmfa.st/v1/models`, which is generated from your config and is always
right.

The one thing that can still drift is **the claims in "Why it's quick"** —
prefix caching, 429 instead of queueing, unbuffered SSE. Those describe what the
gateway actually does today. If you change that behaviour, change the copy.

The model card marks are drawn by us rather than taken from the labs' logos.
Naming a model you serve is ordinary descriptive use; reproducing someone's
trademark on your own marketing implies an endorsement you do not have.
