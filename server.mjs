import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(__dirname, "public");
const PORT = process.env.PORT || 3000;
// Live app runs on Sonnet 5.5 only (never Fable/Opus).
const MODEL = "claude-sonnet-5-5";
const MAX_CONTINUES = 6;

const client = new Anthropic();

const SYSTEM = `You are a competitive-intelligence analyst producing a founder-facing teardown of two Indian health-food / clean-label brands in one product category.

Research rules:
- Use web search to find REAL, CURRENT facts. Search for each brand's founding year, ownership/funding, price and pack size for a comparable SKU, distribution channels, and marketplace ratings/review counts.
- Ingredient honesty is critical: find each brand's actual ingredient panel / label (brand site product page, Amazon.in / BigBasket / Blinkit listing, FSSAI-style label text, OpenFoodFacts). Do NOT rely on marketing copy for this.
- If you cannot find a verifiable ingredient panel for a brand, set ingredient_panel to null, verdict to "unverified", and say so plainly in the explanation. Never guess or reconstruct an ingredient list.
- If any other fact is not verifiable, use null and say "Not verified" in the related note. Never invent numbers, ratings, funding amounts or dates. Prefer values from the source over memory.
- Money is in INR. State the date or source context for prices and ratings when known, since they change.
- Section 8 (takeaway) is advice for a FOUNDER deciding how to position a NEW product in this category. It is not shopper advice; do not say which brand to buy.

Product lock (critical): the user message gives you ONE locked, specific product (SKU) per brand. Compare exactly those products and nothing else.
- Take price, pack size, protein/spec, ingredient panel, rating and review count from the locked product's listing (the retailer and URL given) or, failing that, from the same named product on another retailer or the brand site. Never substitute a different flavour, size, pack count or product line because it surfaced in search.
- Pack counts matter: compute the hero metric per single unit of the locked product, and show the maths.
- If the locked product cannot be found or verified at all, you may use the closest equivalent, but you MUST set "switched": true and explain why in "switch_reason", and print the replacement's exact name. Otherwise "switched" is false and "product_name" is the locked name verbatim.
- Ratings and review counts must come from the locked product's listing; state which retailer.

Output: after researching, reply with ONE JSON object and nothing else (no markdown fences, no prose). Use exactly this shape:

{
  "category": string,
  "brands": [ {"name": string}, {"name": string} ],
  "products": [   // exactly two, brand A then brand B
    {"brand": string, "product_name": string, "retailer": string|null, "listing_url": string|null, "price_inr": number|null, "price_basis": string, "switched": boolean, "switch_reason": string|null}
  ],
  "positioning": {
    "brand_a": {"play": string},   // 1-2 sentences: who they target, what they stand for
    "brand_b": {"play": string},
    "contrast": string             // 2-3 sentences on how the plays differ
  },
  "growth": {
    "brand_a": {"founded": string|null, "ownership": string, "funding": string, "note": string},
    "brand_b": {"founded": string|null, "ownership": string, "funding": string, "note": string}
  },
  "channels": {
    "brand_a": {"summary": string, "channels": [string]},
    "brand_b": {"summary": string, "channels": [string]}
  },
  "hero_metric": {
    "name": string,                // e.g. "Price per 10g protein" - the single most comparable spec for the category
    "unit": string,                // e.g. "₹ per 10g protein"
    "brand_a": {"value": number|null, "display": string, "basis": string},  // basis shows the maths: price, pack size, spec
    "brand_b": {"value": number|null, "display": string, "basis": string},
    "takeaway": string
  },
  "claims_vs_reality": {
    "brand_a": {"headline_claim": string, "ingredient_panel": string|null, "verdict": "match"|"partial"|"mismatch"|"unverified", "explanation": string},
    "brand_b": {"headline_claim": string, "ingredient_panel": string|null, "verdict": "match"|"partial"|"mismatch"|"unverified", "explanation": string}
  },
  "sentiment": {
    "brand_a": {"platform": string, "rating": number|null, "review_count": number|null, "note": string},
    "brand_b": {"platform": string, "rating": number|null, "review_count": number|null, "note": string}
  },
  "white_space": {"summary": string, "gap": string},
  "takeaway": {"summary": string, "moves": [string]},   // 3 concrete positioning moves for a founder
  "data_gaps": [string]            // anything you could not verify
}`;

function extractJson(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error("Model did not return JSON");
  return JSON.parse(text.slice(start, end + 1));
}

// One web-search conversation; resumes pause_turn; returns final text + sources seen.
async function searchChat({ system, user, maxUses, onStatus, sources }) {
  const messages = [{ role: "user", content: user }];
  let searches = 0;
  let message;
  for (let i = 0; i <= MAX_CONTINUES; i++) {
    const stream = client.messages.stream({
      model: MODEL,
      max_tokens: 32000,
      thinking: { type: "adaptive" },
      output_config: { effort: "medium" },
      system,
      tools: [{ type: "web_search_20260209", name: "web_search", max_uses: maxUses }],
      messages,
    });
    stream.on("streamEvent", (ev) => {
      if (ev.type === "content_block_start" && ev.content_block.type === "server_tool_use") {
        searches++;
        onStatus(`Searching the web (${searches})…`);
      }
    });
    message = await stream.finalMessage();
    for (const b of message.content) {
      if (b.type === "web_search_tool_result" && Array.isArray(b.content)) {
        for (const r of b.content) if (r.url && !sources.has(r.url)) sources.set(r.url, r.title || r.url);
      }
    }
    if (message.stop_reason === "pause_turn") {
      messages.push({ role: "assistant", content: message.content });
      onStatus("Still researching…");
      continue;
    }
    break;
  }
  if (message.stop_reason === "refusal") throw new Error("The model declined this request.");
  if (message.stop_reason === "pause_turn") throw new Error("Research took too long; please try again.");
  return message.content.filter((b) => b.type === "text").map((b) => b.text).join("");
}

// ---- Phase 1: commit to one named SKU per brand, persisted so reruns compare the same products ----
const SKU_FILE = path.join(__dirname, "data", "skus.json");
const norm = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
const pairKey = (a, b, cat) => [norm(cat), ...[norm(a), norm(b)].sort()].join("|");
function loadSkus() {
  try { return JSON.parse(fs.readFileSync(SKU_FILE, "utf8")); } catch { return {}; }
}
function saveSkus(db) {
  fs.mkdirSync(path.dirname(SKU_FILE), { recursive: true });
  fs.writeFileSync(SKU_FILE, JSON.stringify(db, null, 2));
}

const SELECT_SYSTEM = `You choose which exact products to compare for a competitor teardown of two Indian brands in one category.

Pick ONE specific, named product (SKU) per brand, so that the two are genuinely comparable:
- Same product tier/spec across brands (e.g. both ~10g-protein, ~50g single bars), and the brand's standard, mainstream, widely stocked version - not a limited edition, combo or variety pack.
- A single named flavour/variant with a unit size, written exactly as it appears on a real listing, e.g. "Yoga Bar Daily 10g Protein Bar, Dark Chocolate & Cranberry, 50g". Set "unit_size" to the size of ONE unit.
- Prefer a listing on Amazon.in (otherwise another major Indian retailer or the brand site) that you actually saw via search, and give its URL. Pick the best-known, most-reviewed qualifying listing.
- Never invent a product name. If you cannot verify a real listing for a brand, set product_name to null.

Reply with ONE JSON object, no prose, no markdown fences:
{"comparison_basis": string, "products": [{"brand": string, "product_name": string|null, "unit_size": string, "spec_per_unit": string, "retailer": string, "listing_url": string|null, "why_chosen": string}, {...}]}
Products must be in the order Brand A, Brand B.`;

async function selectProducts({ brandA, brandB, category }, onStatus, sources) {
  const key = pairKey(brandA, brandB, category);
  const db = loadSkus();
  const hit = db[key];
  if (hit) {
    // order products as the user typed the brands
    const byBrand = new Map(hit.products.map((p) => [norm(p.brand), p]));
    const products = [byBrand.get(norm(brandA)), byBrand.get(norm(brandB))];
    if (products.every(Boolean)) return { ...hit, products, reused: true };
  }
  onStatus("Choosing the exact products to compare…");
  const text = await searchChat({
    system: SELECT_SYSTEM,
    user: `Brand A: ${brandA}\nBrand B: ${brandB}\nCategory: ${category}`,
    maxUses: 6,
    onStatus,
    sources,
  });
  const sel = extractJson(text);
  const products = sel.products;
  if (!Array.isArray(products) || products.length !== 2) throw new Error("Could not choose products to compare.");
  products.forEach((p, i) => {
    p.brand = i === 0 ? brandA : brandB;
    if (!p.product_name) throw new Error(`Could not find a verifiable product listing for ${p.brand}. Try a more specific brand name or category.`);
  });
  const rec = { comparison_basis: sel.comparison_basis, products, selected_at: new Date().toISOString() };
  db[key] = rec;
  saveSkus(db);
  return { ...rec, reused: false };
}

async function generateReport({ brandA, brandB, category }, onStatus) {
  const sources = new Map();
  const sel = await selectProducts({ brandA, brandB, category }, onStatus, sources);
  const lock = sel.products
    .map((p, i) => `Brand ${"AB"[i]} (${p.brand}) LOCKED PRODUCT: "${p.product_name}" - unit size ${p.unit_size}; spec ${p.spec_per_unit}; reference listing: ${p.retailer}${p.listing_url ? " " + p.listing_url : ""}`)
    .join("\n");
  onStatus(sel.reused ? "Reusing the previously chosen products…" : "Products chosen. Researching…");
  const text = await searchChat({
    system: SYSTEM,
    user: `Brand A: ${brandA}\nBrand B: ${brandB}\nCategory: ${category}\n\nComparison basis: ${sel.comparison_basis}\n${lock}\n\nResearch both brands and produce the teardown JSON for exactly these two products.`,
    maxUses: 12,
    onStatus,
    sources,
  });
  onStatus("Assembling report…");
  const report = extractJson(text);

  // Server-side guard: the report must compare the locked products or openly say it switched.
  const rep = Array.isArray(report.products) ? report.products : [];
  report.products = sel.products.map((lp, i) => {
    const used = rep[i] || {};
    const same = norm(used.product_name) === norm(lp.product_name);
    const switched = !same || used.switched === true;
    return {
      brand: lp.brand,
      locked_name: lp.product_name,
      product_name: used.product_name || lp.product_name,
      unit_size: lp.unit_size,
      retailer: used.retailer ?? lp.retailer,
      listing_url: used.listing_url ?? lp.listing_url,
      price_inr: used.price_inr ?? null,
      price_basis: used.price_basis || "",
      switched,
      switch_reason: switched
        ? used.switch_reason ||
          (same
            ? "The report marked this product as switched but gave no reason."
            : "The report used a different product name than the one locked for this comparison, and gave no reason.")
        : null,
    };
  });
  report.product_selection = {
    reused: sel.reused,
    selected_at: sel.selected_at,
    comparison_basis: sel.comparison_basis,
  };
  report.sources = [...sources].slice(0, 30).map(([url, title]) => ({ url, title }));
  report.generated_at = new Date().toISOString();
  report.model = MODEL;
  return report;
}

// ---- jobs & rate limiting (in-memory) ----
const jobs = new Map();
const hits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < 3600_000);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > 10;
}
setInterval(() => {
  const cut = Date.now() - 3600_000;
  for (const [id, j] of jobs) if (j.created < cut) jobs.delete(id);
}, 600_000).unref();

function clean(s, max) {
  return typeof s === "string" ? s.replace(/[\u0000-\u001f]/g, " ").trim().slice(0, max) : "";
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let d = "";
    req.on("data", (c) => {
      d += c;
      if (d.length > 10_000) reject(new Error("Body too large"));
    });
    req.on("end", () => resolve(d));
  });
}

const TYPES = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml" };

const server = http.createServer(async (req, res) => {
  const json = (code, obj) => {
    res.writeHead(code, { "content-type": "application/json" });
    res.end(JSON.stringify(obj));
  };
  try {
    const url = new URL(req.url, "http://x");
    if (req.method === "POST" && url.pathname === "/api/report") {
      const ip = req.socket.remoteAddress || "?";
      if (rateLimited(ip)) return json(429, { error: "Too many reports this hour. Try again later." });
      let body;
      try { body = JSON.parse(await readBody(req)); } catch { return json(400, { error: "Invalid request" }); }
      const brandA = clean(body.brandA, 80), brandB = clean(body.brandB, 80), category = clean(body.category, 80);
      if (!brandA || !brandB || !category) return json(400, { error: "Brand A, Brand B and category are required." });
      if (brandA.toLowerCase() === brandB.toLowerCase()) return json(400, { error: "Pick two different brands." });
      if (!process.env.ANTHROPIC_API_KEY) return json(500, { error: "Server is missing ANTHROPIC_API_KEY." });
      const id = randomUUID();
      const job = { id, created: Date.now(), status: "running", message: "Starting research…" };
      jobs.set(id, job);
      generateReport({ brandA, brandB, category }, (m) => (job.message = m))
        .then((r) => { job.status = "done"; job.report = r; })
        .catch((e) => {
          console.error(e);
          job.status = "error";
          job.error = e instanceof Anthropic.APIError ? `API error (${e.status}): ${e.message}` : e.message;
        });
      return json(202, { id });
    }
    const m = url.pathname.match(/^\/api\/report\/([\w-]+)$/);
    if (req.method === "GET" && m) {
      const job = jobs.get(m[1]);
      if (!job) return json(404, { error: "Report not found (it may have expired)." });
      return json(200, { status: job.status, message: job.message, report: job.report, error: job.error });
    }
    if (req.method === "GET") {
      const file = url.pathname === "/" || url.pathname === "/report" ? "/index.html" : url.pathname;
      const p = path.normalize(path.join(PUBLIC, file));
      if (p.startsWith(PUBLIC) && fs.existsSync(p) && fs.statSync(p).isFile()) {
        res.writeHead(200, { "content-type": TYPES[path.extname(p)] || "application/octet-stream" });
        return fs.createReadStream(p).pipe(res);
      }
    }
    json(404, { error: "Not found" });
  } catch (e) {
    console.error(e);
    json(500, { error: "Server error" });
  }
});

server.listen(PORT, () => console.log(`Teardown on http://localhost:${PORT}`));
