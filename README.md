# Teardown

Self-serve competitor teardown for two Indian health-food / clean-label brands.
Enter two brands and a category; the server researches the web with Claude (`claude-sonnet-5-5` + web search) and renders an 8-section report you can print to PDF.

## Run
    npm install
    TEARDOWN_API_KEY=... npm start   # http://localhost:3000 (ANTHROPIC_API_KEY also works as a fallback)

The product pair chosen for each comparison is saved in `data/skus.json` so reruns compare the same products.

## Configuration

| Variable | Purpose |
|---|---|
| `TEARDOWN_API_KEY` | Anthropic API key (`ANTHROPIC_API_KEY` works as a fallback locally) |
| `TEARDOWN_PASSWORD` | Shared password (HTTP Basic auth, any username). Required when `NODE_ENV=production`; if unset locally, protection is off |
| `REPORTS_PER_HOUR` | Report cap, default 10. Behind a proxy this is effectively one global cap |
| `PORT` | Set by Render; defaults to 3000 |

## Deploy to Render

`render.yaml` is a Blueprint: Render creates the service from it and asks you for the two secrets.

Known limits of the free plan (see Render's docs): it spins down after 15 minutes idle and takes about a minute to wake;
the filesystem is wiped on every restart, redeploy and spin-down, so `data/skus.json` (the locked products, prices and ratings)
and any in-progress report disappear with it. Fine for a first test; move to a paid plan and a persistent disk before real use.
