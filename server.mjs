import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createHash, randomUUID, timingSafeEqual } from "node:crypto";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(__dirname, "public");
const PORT = process.env.PORT || 3000;
// Live app runs on Sonnet 5.5 only (never Fable/Opus).
const MODEL = "claude-sonnet-5-5";
const MAX_CONTINUES = 6;
const SELECT_MAX_SEARCHES = 24; // product selection: verify price, pack and spec for two brands
const RESEARCH_MAX_SEARCHES = 12;
const SENTIMENT_MAX_SEARCHES = 12; // one-off rating capture for the locked products

// The deployment secret is TEARDOWN_API_KEY; ANTHROPIC_API_KEY is only a fallback for local use.
const API_KEY = process.env.TEARDOWN_API_KEY || process.env.ANTHROPIC_API_KEY;
const client = new Anthropic({ apiKey: API_KEY });

// One shared password (HTTP Basic auth; the username is ignored). Good enough for testing, not a user system.
const PASSWORD = process.env.TEARDOWN_PASSWORD;
if (!PASSWORD) {
  if (process.env.NODE_ENV === "production") {
    console.error("Refusing to start: TEARDOWN_PASSWORD is not set (NODE_ENV=production).");
    process.exit(1);
  }
  console.warn("TEARDOWN_PASSWORD is not set: password protection is OFF (local use only).");
}
const sha = (v) => createHash("sha256").update(v).digest();
function passwordOk(req) {
  const h = req.headers.authorization || "";
  if (!h.startsWith("Basic ")) return false;
  const decoded = Buffer.from(h.slice(6), "base64").toString("utf8");
  const given = decoded.slice(decoded.indexOf(":") + 1);
  return timingSafeEqual(sha(given), sha(PASSWORD));
}
// Failed-attempt cap (global, so it can't be dodged by spoofing an IP): 30 failures per 15 minutes.
let failures = [];
const lockedOut = () => {
  const cut = Date.now() - 900_000;
  failures = failures.filter((t) => t > cut);
  return failures.length >= 30;
};
const REPORTS_PER_HOUR = Number(process.env.REPORTS_PER_HOUR) || 10;

// Every string that can reach the report is written in impersonal, third-person analytical voice.
const VOICE = `Voice (applies to EVERY string you write, including notes, explanations and reasons): impersonal, third-person, analytical. Never use first person ("I", "I'm", "I've", "my", "me", "we", "our") and never refer to yourself, your searches or your process ("my search", "I found", "I could not", "not directly visible to me", "I did not check"). Put uncertainty on the DATA, not on yourself:
- "I could not verify X" -> "X is unconfirmed" or "X could not be independently verified"
- "I did not find a rating" -> "No rating was located for this listing"
- "not visible to me" -> "not shown on the listing"
- "I took the figure from..." -> "The figure is taken from..."
State findings as facts about the brands and products.

Format: plain text only in every string. No HTML or XML tags (never <cite> or any citation markup), no markdown. Name sources in plain words (for example "Amazon.in listing").`;

// (SYSTEM, SELECT_SYSTEM, SENTIMENT_SYSTEM and INFER_SYSTEM below all include VOICE.)
const SYSTEM = `You are a competitive-intelligence analyst producing a founder-facing teardown of two Indian health-food / clean-label brands in one product category.

${VOICE}

Research rules:
- Use web search to find REAL, CURRENT facts. Search for each brand's founding year, ownership/funding, price and pack size for a comparable SKU, distribution channels, and marketplace ratings/review counts.
- Ingredient honesty is critical: find each brand's actual ingredient panel / label (brand site product page, Amazon.in / BigBasket / Blinkit listing, FSSAI-style label text, OpenFoodFacts). Do NOT rely on marketing copy for this.
- If you cannot find a verifiable ingredient panel for a brand, set ingredient_panel to null, verdict to "unverified", and say so plainly in the explanation. Never guess or reconstruct an ingredient list.
- If any other fact is not verifiable, use null and say "Not verified" in the related note. Never invent numbers, ratings, funding amounts or dates. Prefer values from the source over memory.
- Money is in INR. State the date or source context for prices and ratings when known, since they change.
- Section 8 (takeaway) is advice for a FOUNDER deciding how to position a NEW product in this category. It is not shopper advice; do not say which brand to buy.

Product lock (critical): the user message gives you ONE locked, specific product (SKU) per brand. Compare exactly those products and nothing else.
- Take price, pack size, protein/spec, ingredient panel, rating and review count from the locked product's listing (the retailer and URL given) or, failing that, from the same named product on another retailer or the brand site. Never substitute a different flavour, size, pack count or product line because it surfaced in search.
- The locked product's price, pack count and per-unit spec are fixed inputs, already verified. The server computes the hero metric from them. Do NOT recompute it, do NOT report a different price or spec, and do NOT put any hero-metric number in hero_metric - only the "takeaway" prose, which may quote the computed values given in the user message exactly.
- If the locked product cannot be found or verified at all, you may use the closest equivalent, but you MUST set "switched": true and explain why in "switch_reason", and print the replacement's exact name. Otherwise "switched" is false and "product_name" is the locked name verbatim.
- Ratings and review counts are captured and locked by the server. Do NOT research, report or mention them (and do not list them in data_gaps).

Output: after researching, reply with ONE JSON object and nothing else (no markdown fences, no prose). Use exactly this shape:

{
  "category": string,
  "brands": [ {"name": string}, {"name": string} ],
  "products": [   // exactly two, brand A then brand B
    {"brand": string, "product_name": string, "retailer": string|null, "listing_url": string|null, "switched": boolean, "switch_reason": string|null}
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
    "takeaway": string             // 1-2 sentences comparing the two computed values given in the user message; quote them exactly
  },
  "claims_vs_reality": {
    "brand_a": {"headline_claim": string, "ingredient_panel": string|null, "verdict": "match"|"partial"|"mismatch"|"unverified", "explanation": string},
    "brand_b": {"headline_claim": string, "ingredient_panel": string|null, "verdict": "match"|"partial"|"mismatch"|"unverified", "explanation": string}
  },
  "white_space": {"summary": string, "gap": string},
  "takeaway": {"summary": string, "moves": [string]},   // 3 concrete positioning moves for a founder
  "data_gaps": [string]            // anything you could not verify
}`;

// Web-search citations can leak into text as <cite index="..">...</cite>. Keep the cited text, drop all markup.
const TAG_TEST = /<\/?[a-zA-Z]/;
const TAG_RE = /<\/?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*)?\/?>/g;
function cleanStrings(node, stat = { changed: 0 }) {
  const fix = (v) => {
    if (typeof v !== "string" || !TAG_TEST.test(v)) return v;
    const out = v.replace(TAG_RE, "").replace(/[ \t]{2,}/g, " ").trim();
    if (out !== v) stat.changed++;
    return out;
  };
  if (Array.isArray(node)) node.forEach((v, i) => (typeof v === "string" ? (node[i] = fix(v)) : cleanStrings(v, stat)));
  else if (node && typeof node === "object") for (const [k, v] of Object.entries(node)) typeof v === "string" ? (node[k] = fix(v)) : cleanStrings(v, stat);
  return stat;
}

// Every balanced {...} block in the text (string-aware), so a stray brace in surrounding prose can't break parsing.
function jsonCandidates(text) {
  const out = [];
  for (let i = text.indexOf("{"); i >= 0; i = text.indexOf("{", i + 1)) {
    let depth = 0, inStr = false, escaped = false;
    for (let j = i; j < text.length; j++) {
      const c = text[j];
      if (inStr) { if (escaped) escaped = false; else if (c === "\\") escaped = true; else if (c === '"') inStr = false; continue; }
      if (c === '"') inStr = true;
      else if (c === "{") depth++;
      else if (c === "}" && --depth === 0) { out.push(text.slice(i, j + 1)); break; }
    }
  }
  return out;
}

function extractJson(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error("Model did not return JSON");
  let obj;
  let path = "direct";
  try {
    obj = JSON.parse(text.slice(start, end + 1));
  } catch {
    path = "fallback";
    // Fall back to the largest block that parses on its own (the report is always the biggest object).
    for (const c of jsonCandidates(text).sort((a, b) => b.length - a.length)) {
      try { obj = JSON.parse(c); break; } catch { /* try the next one */ }
    }
    if (!obj) {
      console.error("Unparseable model reply. Start:", text.slice(0, 300), "... End:", text.slice(-300));
      throw new Error("Model did not return valid JSON; please try again.");
    }
  }
  const { changed } = cleanStrings(obj);
  console.log(`[json] parse=${path} chars=${text.length} markup_stripped_from=${changed}_strings`);
  return obj;
}

// One web-search conversation; resumes pause_turn; returns final text + sources seen.
async function searchChat({ system, user, maxUses, onStatus, sources }) {
  const messages = [{ role: "user", content: user }];
  let searches = 0;
  let limitHit = false;
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
      if (b.type !== "web_search_tool_result") continue;
      if (Array.isArray(b.content)) {
        for (const r of b.content) if (r.url && !sources.has(r.url)) sources.set(r.url, r.title || r.url);
      } else if (b.content?.error_code === "max_uses_exceeded") limitHit = true;
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
  const text = message.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  return { text, searches, limitHit, maxUses };
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

const SELECT_SYSTEM = `You choose which exact products to compare for a competitor teardown of two Indian brands in one category, and you capture the raw numbers for the category's hero metric. You do NOT calculate the metric; a program does that from your raw numbers.

1. Decide the category's single most comparable per-unit spec and the comparison quantity. Examples: protein bars -> spec "protein", unit "g", per 10 ("Price per 10g protein"); muesli -> spec "net weight", unit "g", per 100 ("Price per 100g"); sunscreen -> spec "volume", unit "ml", per 50 ("Price per 50ml").
2. Pick ONE specific, named product (SKU) per brand, genuinely comparable across the two brands:
   - Same tier/spec (e.g. both ~10g-protein, ~50g single bars) and the brand's standard, mainstream, widely stocked version - not a limited edition, combo or variety pack.
   - A single named flavour/variant, with the name written exactly as on a real listing, e.g. "Yoga Bar Daily 10g Protein Bar, Dark Chocolate & Cranberry, 50g".
   - Prefer Amazon.in (otherwise another major Indian retailer or the brand site). Pick the best-known, most-reviewed qualifying listing.
3. Capture raw numbers for each product, all from the SAME listing where possible:
   - price_inr: the selling price shown on that listing, as a plain number in rupees.
   - pack_count: how many single units that price buys (a 6-pack at 415 -> pack_count 6; a single bar -> 1).
   - spec_value: the amount of the spec in ONE single unit, from the label or listing (e.g. 10 for 10g protein in one bar). Use a printed/label figure, not an estimate.
   - price_source and spec_source: where each number came from, as {"retailer": short site name such as "Amazon.in", "url": the product page URL or null}. No sentences.
Never invent or estimate numbers. Verify price, pack_count and spec_value for both products before answering. If a candidate SKU's protein/spec or price cannot be verified, switch to another SKU of the same brand where all three can be verified. Only if no SKU can be verified, set that product's price_inr or spec_value to null.

${VOICE}

You have a generous search budget (about ${SELECT_MAX_SEARCHES} searches). Work efficiently and stop searching once both products are fully verified.

Reply with ONE JSON object, no prose, no markdown fences:
{"comparison_basis": string,
 "metric": {"name": string, "spec_name": string, "spec_unit": string, "per": number, "unit_label": string},
 "products": [{"brand": string, "product_name": string|null, "unit_size": string, "spec_value": number|null, "spec_source": {"retailer": string, "url": string|null}, "price_inr": number|null, "pack_count": number|null, "price_source": {"retailer": string, "url": string|null}, "retailer": string, "listing_url": string|null, "why_chosen": string}, {...}]}
"unit_label" example: "₹ per 10g protein". Products must be in the order Brand A, Brand B.`;

const SKU_SCHEMA_VERSION = 3; // v2 added price/pack/spec; v3 also locks rating and review count
const posNum = (n) => typeof n === "number" && Number.isFinite(n) && n > 0;

function validateSelection(sel) {
  const m = sel.metric;
  if (!m || !posNum(m.per) || !m.name || !m.unit_label || !m.spec_name || !m.spec_unit) throw new Error("Selection did not define a usable comparison metric.");
  if (!Array.isArray(sel.products) || sel.products.length !== 2) throw new Error("Could not choose products to compare.");
  for (const p of sel.products) {
    if (!p.product_name) throw new Error(`Could not find a verifiable product listing for ${p.brand}. Try a more specific brand name or category.`);
    const missing = [!posNum(p.price_inr) && "price", !posNum(p.pack_count) && "pack count", !posNum(p.spec_value) && m.spec_name].filter(Boolean);
    if (missing.length) throw new Error(`Could not verify ${missing.join(" / ")} for ${p.brand} (${p.product_name}).`);
  }
}

// The hero metric is computed here, in code, from locked raw numbers. The model never states it.
const r2 = (n) => Math.round(n * 100) / 100;

// Turn a source (free text with a URL buried in it, or {retailer,url}) into a short label plus a link.
const SITE_LABELS = [[/amazon\.in$/, "Amazon.in"], [/blinkit\.com$/, "Blinkit"], [/swiggy\.com$/, "Swiggy Instamart"], [/zepto/, "Zepto"], [/bigbasket/, "BigBasket"], [/flipkart/, "Flipkart"], [/jiomart/, "JioMart"], [/nykaa/, "Nykaa"], [/pharmeasy/, "PharmEasy"], [/1mg\.com$/, "Tata 1mg"], [/apollopharmacy/, "Apollo Pharmacy"], [/openfoodfacts/, "OpenFoodFacts"]];
function sourceRef(src, fallbackLabel) {
  let url = null;
  let label = null;
  if (src && typeof src === "object") { url = src.url || null; label = src.retailer || src.label || null; }
  else if (typeof src === "string") {
    const urls = src.match(/https?:\/\/[^\s)\];,'"]+/g) || [];
    url = urls.find((u) => /\/(dp|prn|product|p)\//.test(u)) || urls[0] || null; // prefer a product page over a search page
  }
  if (url && !/^https?:\/\//.test(url)) url = null;
  let host = "";
  try { host = new URL(url).hostname.replace(/^www\./, ""); } catch { /* no url */ }
  const known = SITE_LABELS.find(([re]) => re.test(host))?.[1];
  return { label: known || label || host || fallbackLabel || "source", url };
}

// The hero metric is computed here, in code, from locked raw numbers. The model never states it.
function computeHero(metric, p) {
  const perUnit = p.price_inr / p.pack_count;
  const value = r2((perUnit / p.spec_value) * metric.per);
  const unit = `${metric.per}${metric.spec_unit} ${metric.spec_name}`;
  const basis =
    (p.pack_count === 1 ? `₹${p.price_inr} for 1 unit. ` : `₹${p.price_inr} ÷ ${p.pack_count} units = ₹${r2(perUnit)} per unit. `) +
    `₹${r2(perUnit)} ÷ ${p.spec_value}${metric.spec_unit} ${metric.spec_name} × ${metric.per} = ₹${value} per ${unit}.`;
  return {
    value,
    display: `₹${value.toFixed(2)}`,
    basis,
    price_ref: sourceRef(p.price_source, p.retailer),
    spec_ref: { ...sourceRef(p.spec_source, p.retailer), what: metric.spec_name },
  };
}

// ---- Voice enforcement: rewrite any first-person sentence that slips through, once, before it is saved/shown ----
const FIRST_PERSON = [/\bI(?:'(?:m|ve|d|ll))?\b/, /\b(?:me|my|mine|myself|we|our|ours)\b/i];
const VOICE_SKIP_KEYS = new Set(["headline_claim", "ingredient_panel", "product_name", "locked_name", "listing_url", "source_url", "url", "title"]); // verbatim quotes, names, links
const hasFirstPerson = (t) => !/^https?:\/\//.test(t) && FIRST_PERSON.some((re) => re.test(t));
function collectVoiceTargets(node, out = []) {
  const visit = (parent, key, v) => {
    if (typeof v === "string") { if (hasFirstPerson(v)) out.push({ parent, key, text: v }); }
    else if (v && typeof v === "object") collectVoiceTargets(v, out);
  };
  if (Array.isArray(node)) node.forEach((v, i) => visit(node, i, v));
  else if (node && typeof node === "object") for (const [k, v] of Object.entries(node)) if (!VOICE_SKIP_KEYS.has(k)) visit(node, k, v);
  return out;
}
const REWRITE_SYSTEM = `Rewrite each string in the JSON array you are given into impersonal, third-person analytical voice. Remove every first-person word (I, my, me, we, our) and every reference to the writer's own searching or process. Keep every fact, number, price, name and URL exactly as written. Examples: "I could not verify the founding year" -> "The founding year could not be independently verified."; "I took the 6,549 from the page" -> "The figure of 6,549 is taken from the page."; "not visible to me" -> "not shown on the listing". Reply with ONLY a JSON array of the rewritten strings, same length and order, no prose, no markdown.`;
async function impersonalise(obj) {
  const targets = collectVoiceTargets(obj);
  if (!targets.length) return { rewritten: 0, remaining: 0 };
  let rewritten = 0;
  try {
    const msg = await client.messages.create({
      model: MODEL,
      max_tokens: 16000,
      output_config: { effort: "low" },
      system: REWRITE_SYSTEM,
      messages: [{ role: "user", content: JSON.stringify(targets.map((t) => t.text)) }],
    });
    const raw = msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
    const arr = JSON.parse(raw.slice(raw.indexOf("["), raw.lastIndexOf("]") + 1));
    if (Array.isArray(arr) && arr.length === targets.length) {
      targets.forEach((t, i) => {
        if (typeof arr[i] === "string" && arr[i].trim() && !hasFirstPerson(arr[i])) { t.parent[t.key] = arr[i]; rewritten++; }
      });
    }
  } catch (e) {
    console.error("voice rewrite failed:", e.message);
  }
  return { rewritten, remaining: collectVoiceTargets(obj).length };
}

// ---- Category inference (only when the user leaves the category blank) ----
const INFER_MAX_SEARCHES = 8;
const INFER_SYSTEM = `You check whether two Indian consumer brands share ONE clear product category in which their products can be compared like-for-like.

Use web search briefly to identify what each brand mainly sells in India. Then decide:
- "shared_category": a specific product category both brands actually sell, in the plural, as a shopper would search it (e.g. "protein bars", "muesli", "sunscreen"). Broad umbrellas such as "health food", "snacks", "wellness" or "FMCG" do NOT count: a per-unit price comparison must make sense.
- If the brands mainly sell different kinds of products (e.g. a protein-bar brand and a sunscreen brand), or either brand cannot be found, or it is a judgement call between several categories, set "shared_category" to null. Prefer null over a guess.

${VOICE}

Reply with ONE JSON object, no prose, no markdown fences:
{"shared_category": string|null, "brand_a_sells": string, "brand_b_sells": string, "reason": string}`;

async function inferCategory({ brandA, brandB }, onStatus, sources) {
  const key = "infer|" + [norm(brandA), norm(brandB)].sort().join("|");
  const db = loadSkus();
  if (db[key]?.category) return db[key].category; // keep the inferred category stable across runs
  onStatus("No category given. Checking what the two brands have in common…");
  const { text } = await searchChat({
    system: INFER_SYSTEM,
    user: `Brand A: ${brandA}\nBrand B: ${brandB}`,
    maxUses: INFER_MAX_SEARCHES,
    onStatus,
    sources,
  });
  const r = extractJson(text);
  const cat = typeof r.shared_category === "string" ? r.shared_category.trim().slice(0, 80) : "";
  if (!cat) {
    await impersonalise(r);
    throw new Error(
      `No clear shared product category was found for ${brandA} and ${brandB}` +
        (r.brand_a_sells && r.brand_b_sells ? ` (${brandA}: ${r.brand_a_sells}; ${brandB}: ${r.brand_b_sells})` : "") +
        `. Please specify a category (for example "protein bars" or "muesli") and try again.`,
    );
  }
  db[key] = { category: cat, brand_a_sells: r.brand_a_sells, brand_b_sells: r.brand_b_sells, inferred_at: new Date().toISOString() };
  saveSkus(db);
  return cat;
}

const SENTIMENT_SYSTEM = `You record public buyer ratings for two specific, already-chosen products. Use web search.

For each product find the star rating (out of 5) and the number of ratings/reviews shown for THAT product's listing. Use the reference listing given; if it isn't reachable, use the same named product on another major Indian retailer and say which. Copy the numbers as displayed; never estimate, average across different products, or substitute another flavour, size or pack.
- review_count is the count displayed next to the rating (ratings or reviews, say which in "note").
- If you cannot verify a rating or count for a product, set it to null and explain in "note". Do not guess.
Stop searching once both are found or clearly unavailable.

${VOICE}

Reply with ONE JSON object, no prose, no markdown fences:
{"sentiment": [{"brand": string, "platform": string, "rating": number|null, "review_count": number|null, "source_url": string|null, "note": string}, {...}]}
Order: Brand A, Brand B.`;

// Clamp to sane values so a junk number can't be locked in: anything unverifiable becomes null ("not verified").
function cleanSentiment(x = {}) {
  const rating = typeof x.rating === "number" && x.rating > 0 && x.rating <= 5 ? x.rating : null;
  const rc = typeof x.review_count === "number" && Number.isInteger(x.review_count) && x.review_count >= 0 ? x.review_count : null;
  return {
    platform: x.platform || null,
    rating,
    review_count: rc,
    source_url: x.source_url || null,
    note: x.note || "",
  };
}

async function captureSentiment(rec, onStatus, sources) {
  onStatus("Recording buyer ratings for the locked products…");
  const user = rec.products
    .map((p, i) => `Brand ${"AB"[i]} (${p.brand}): "${p.product_name}" - reference listing: ${p.retailer}${p.listing_url ? " " + p.listing_url : ""}`)
    .join("\n");
  const { text, searches, limitHit } = await searchChat({
    system: SENTIMENT_SYSTEM,
    user,
    maxUses: SENTIMENT_MAX_SEARCHES,
    onStatus,
    sources,
  });
  const parsed = extractJson(text);
  const got = Array.isArray(parsed.sentiment) ? parsed.sentiment : [];
  rec.sentiment = rec.products.map((_, i) => cleanSentiment(got[i]));
  rec.sentiment_captured_at = new Date().toISOString();
  rec.sentiment_searches = searches;
  rec.sentiment_limit_hit = limitHit;
}

async function selectProducts({ brandA, brandB, category }, onStatus, sources) {
  const key = pairKey(brandA, brandB, category);
  const db = loadSkus();
  const hit = db[key];
  if (hit && hit.v >= 2) {
    if (cleanStrings(hit).changed) saveSkus(db); // strip any markup saved by older runs
    if (hit.v >= SKU_SCHEMA_VERSION && (await impersonalise(hit)).rewritten) saveSkus(db); // clean older first-person notes once
    if (hit.v < SKU_SCHEMA_VERSION) {
      // Older lock: keep the same products, capture sentiment once and save it.
      await captureSentiment(hit, onStatus, sources);
      await impersonalise(hit);
      hit.v = SKU_SCHEMA_VERSION;
      saveSkus(db);
    }
    // order products (and their locked sentiment) as the user typed the brands
    const idx = new Map(hit.products.map((p, i) => [norm(p.brand), i]));
    const order = [idx.get(norm(brandA)), idx.get(norm(brandB))];
    if (order.every((i) => i !== undefined)) {
      return { ...hit, products: order.map((i) => hit.products[i]), sentiment: order.map((i) => hit.sentiment[i]), reused: true };
    }
  }
  onStatus("Choosing the exact products to compare…");
  const { text, searches, limitHit } = await searchChat({
    system: SELECT_SYSTEM,
    user: `Brand A: ${brandA}\nBrand B: ${brandB}\nCategory: ${category}`,
    maxUses: SELECT_MAX_SEARCHES,
    onStatus,
    sources,
  });
  const sel = extractJson(text);
  if (Array.isArray(sel.products)) sel.products.forEach((p, i) => (p.brand = i === 0 ? brandA : brandB));
  validateSelection(sel);
  const rec = {
    v: 2, // upgraded to the current version once sentiment is captured

    comparison_basis: sel.comparison_basis,
    metric: sel.metric,
    products: sel.products,
    selected_at: new Date().toISOString(),
    selection_searches: searches,
    selection_limit_hit: limitHit,
  };
  await impersonalise(rec);
  db[key] = rec;
  saveSkus(db); // keep the chosen products even if the sentiment step fails; the next run finishes the upgrade
  await captureSentiment(rec, onStatus, sources);
  await impersonalise(rec);
  rec.v = SKU_SCHEMA_VERSION;
  saveSkus(db);
  return { ...rec, reused: false };
}

async function generateReport({ brandA, brandB, category }, onStatus) {
  const sources = new Map();
  const categoryInferred = !category;
  if (categoryInferred) category = await inferCategory({ brandA, brandB }, onStatus, sources);
  const sel = await selectProducts({ brandA, brandB, category }, onStatus, sources);
  const hero = sel.products.map((p) => computeHero(sel.metric, p));
  const lock = sel.products
    .map((p, i) => `Brand ${"AB"[i]} (${p.brand}) LOCKED PRODUCT: "${p.product_name}" - unit size ${p.unit_size}; locked price ₹${p.price_inr} for ${p.pack_count} unit(s); ${sel.metric.spec_name} ${p.spec_value}${sel.metric.spec_unit} per unit; reference listing: ${p.retailer}${p.listing_url ? " " + p.listing_url : ""}\n  Computed hero metric (${sel.metric.unit_label}): ${hero[i].display}`)
    .join("\n");
  onStatus(sel.reused ? "Reusing the previously chosen products…" : "Products chosen. Researching…");
  const research = await searchChat({
    system: SYSTEM,
    user: `Brand A: ${brandA}\nBrand B: ${brandB}\nCategory: ${category}\n\nComparison basis: ${sel.comparison_basis}\n${lock}\n\nResearch both brands and produce the teardown JSON for exactly these two products.`,
    maxUses: RESEARCH_MAX_SEARCHES,
    onStatus,
    sources,
  });
  onStatus("Assembling report…");
  const report = extractJson(research.text);

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
      locked_price_inr: lp.price_inr,
      locked_pack_count: lp.pack_count,
      locked_spec_value: lp.spec_value,
      switched,
      switch_reason: switched
        ? used.switch_reason ||
          (same
            ? "The report marked this product as switched but gave no reason."
            : "The report used a different product name than the one locked for this comparison, and gave no reason.")
        : null,
    };
  });

  // Hero metric: always from the locked raw numbers, never from model text. If a product was switched,
  // the locked price/spec no longer describe what was compared, so show no number instead of a wrong one.
  const heroFor = (i) =>
    report.products[i].switched
      ? { value: null, display: "n/a", basis: `Not computed: the report switched products, and the locked price and ${sel.metric.spec_name} belong to "${report.products[i].locked_name}".` }
      : hero[i];
  report.hero_metric = {
    name: sel.metric.name,
    unit: sel.metric.unit_label,
    brand_a: heroFor(0),
    brand_b: heroFor(1),
    takeaway: typeof report.hero_metric?.takeaway === "string" ? report.hero_metric.takeaway : "",
  };

  // Sentiment: always the locked values. Null stays "not verified" every run; a switched product gets none.
  const sentFor = (i) => {
    const ls = sel.sentiment[i] || cleanSentiment();
    if (report.products[i].switched) {
      return { platform: ls.platform, rating: null, review_count: null, note: `Not verified: the report switched products, and the locked rating belongs to "${report.products[i].locked_name}".` };
    }
    const when = new Date(sel.sentiment_captured_at).toISOString().slice(0, 10);
    const note = ls.rating == null && ls.review_count == null
      ? `Not verified: no rating or review count could be confirmed for this listing when it was locked (${when}). ${ls.note}`.trim()
      : `Locked ${when}. ${ls.note}`.trim();
    return { platform: ls.platform, rating: ls.rating, review_count: ls.review_count, note };
  };
  report.sentiment = { brand_a: sentFor(0), brand_b: sentFor(1) };

  report.product_selection = {
    reused: sel.reused,
    selected_at: sel.selected_at,
    comparison_basis: sel.comparison_basis,
    selection_searches: sel.selection_searches ?? null,
    selection_limit_hit: sel.selection_limit_hit ?? null,
    selection_search_limit: SELECT_MAX_SEARCHES,
    sentiment_captured_at: sel.sentiment_captured_at,
    sentiment_searches: sel.sentiment_searches ?? null,
    sentiment_limit_hit: sel.sentiment_limit_hit ?? null,
    research_searches: research.searches,
    research_limit_hit: research.limitHit,
  };
  report.sources = [...sources].slice(0, 30).map(([url, title]) => ({ url, title }));
  report.generated_at = new Date().toISOString();
  report.model = MODEL;
  report.category = category;
  report.category_inferred = categoryInferred;
  report.voice_check = await impersonalise(report); // rewrite any first-person phrasing that slipped through
  return report;
}

// Errors that arrive mid-stream carry no HTTP status; the details are in the JSON body (or message).
const ERROR_TYPE_STATUS = {
  invalid_request_error: 400, authentication_error: 401, permission_error: 403, not_found_error: 404,
  request_too_large: 413, rate_limit_error: 429, api_error: 500, overloaded_error: 529,
};
function describeError(e) {
  if (!(e instanceof Anthropic.APIError)) return e.message;
  let body = e.error;
  if (!body?.error) { try { body = JSON.parse(e.message); } catch { body = null; } }
  const status = e.status ?? ERROR_TYPE_STATUS[body?.error?.type] ?? "no status";
  const id = body?.request_id ? ` [request ${body.request_id}]` : "";
  return `API error (${status}): ${body?.error?.message ?? e.message}${id}`;
}

// ---- jobs & rate limiting (in-memory) ----
const jobs = new Map();
const hits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < 3600_000);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > REPORTS_PER_HOUR;
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
    if (url.pathname === "/healthz") { res.writeHead(200, { "content-type": "text/plain" }); return res.end("ok"); }
    if (PASSWORD) {
      if (lockedOut()) return json(429, { error: "Too many failed password attempts. Try again in 15 minutes." });
      if (!passwordOk(req)) {
        failures.push(Date.now());
        res.writeHead(401, { "www-authenticate": 'Basic realm="Teardown", charset="UTF-8"', "content-type": "text/plain" });
        return res.end("Password required");
      }
    }
    if (req.method === "POST" && url.pathname === "/api/report") {
      const ip = req.socket.remoteAddress || "?";
      if (rateLimited(ip)) return json(429, { error: "Too many reports this hour. Try again later." });
      let body;
      try { body = JSON.parse(await readBody(req)); } catch { return json(400, { error: "Invalid request" }); }
      const brandA = clean(body.brandA, 80), brandB = clean(body.brandB, 80), category = clean(body.category, 80);
      if (!brandA || !brandB) return json(400, { error: "Brand A and Brand B are required." });
      if (brandA.toLowerCase() === brandB.toLowerCase()) return json(400, { error: "Pick two different brands." });
      if (!API_KEY) return json(500, { error: "Server is missing TEARDOWN_API_KEY." });
      const id = randomUUID();
      const job = { id, created: Date.now(), status: "running", message: "Starting research…" };
      jobs.set(id, job);
      generateReport({ brandA, brandB, category }, (m) => (job.message = m))
        .then((r) => { job.status = "done"; job.report = r; })
        .catch((e) => {
          console.error(e);
          job.status = "error";
          job.error = describeError(e);
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
