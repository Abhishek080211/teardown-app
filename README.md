# Teardown

Self-serve competitor teardown for two Indian health-food / clean-label brands.
Enter two brands and a category; the server researches the web with Claude (`claude-sonnet-5-5` + web search) and renders an 8-section report you can print to PDF.

## Run
    npm install
    ANTHROPIC_API_KEY=... npm start   # http://localhost:3000

The product pair chosen for each comparison is saved in `data/skus.json` so reruns compare the same products.
