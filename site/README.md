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

The legal pages are **drafts**. Every field highlighted in yellow is a
placeholder that must be replaced, and both pages carry a banner saying so.
Search for `class="fill"` to find them all, then delete the `.draft-banner`
blocks once a lawyer has reviewed the result.

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

Two things on the home page are hard-coded and will drift:

- **The models table** duplicates prices from `config/config.yaml`. The live
  source is always `api.llmfa.st/v1/models`, which the page links to. Update the
  table whenever you change pricing, or delete it and link out instead.
- **The claims in "Why it's quick"** describe what the gateway actually does
  today: prefix caching, 429 instead of queueing, unbuffered SSE. If you change
  that behaviour, change the copy.
