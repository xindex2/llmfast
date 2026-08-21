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

The legal pages are **drafts**. Everything the domain settles is already filled
in — contact addresses, notice periods, currency. What is still highlighted in
yellow is what only you can supply:

| Field | Where |
|---|---|
| Company legal name, number, registered address | both pages |
| Jurisdiction and venue | terms §25, privacy §1 |
| Effective date | both, at the top |
| Retention choice | terms §9, privacy §3 — keep exactly one |
| Serving region and sub-processors | privacy §8, §9 |
| Website analytics, if any | privacy §6 |

Search for `class="fill"` to find them all, then delete the `.draft-banner`
blocks once a lawyer has reviewed the result.

Company details are deliberately left blank rather than guessed. A fabricated
legal entity in a Terms of Use is worse than an obvious gap.

The two things most likely to get you in trouble:

1. **Retention claims must match reality.** The Privacy Policy offers a zero
   data retention section and a limited retention section. Keep exactly one.
   Whichever you keep must agree with `server.raw_retention_days` in
   `config/config.yaml` and with the `compliance.zdr` flag you publish on every
   model. OpenRouter asks about this directly on the application form.
2. **Jurisdiction.** The governing law, venue and liability cap clauses are
   written generically. They need to name the country you are actually
   registered in, and a lawyer there needs to confirm the caps are enforceable.

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
