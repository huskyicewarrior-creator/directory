/* P7-HS client-side filter engine for static hosting (Vercel).
   Runs fully on the user's device: a deterministic spec compiler (a faithful
   port of app.py heuristic_parse_prompt / merge_specs / apply_filter_spec)
   plus an optional in-browser LLM assist via @mlc-ai/web-llm (WebGPU).
   The deterministic layer answers alone when WebGPU or the model is
   unavailable, so filtering never depends on a download. */

const CLIENT_TEACHERS = [
  "Bernstein", "Bilandzija", "Ervin", "Geckle", "Laska", "Lin", "Luthey",
  "Makhoul", "Mascheroni", "Oshiro", "Staley", "Turnipseed", "Velarde", "Wild",
];

export const BROWSER_MODEL_ID = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
const WEBLLM_CDN = "https://esm.sh/@mlc-ai/web-llm@0.2.79";

const SYSTEM_PROMPT = `You are the P7-HS Roster Query Compiler, a precise student roster query engine.
Given a user's natural language request to filter a school roster, output ONLY a valid JSON object with these keys:
{
  "gender": "boy" | "girl" | null,
  "homeroom_teacher": string or null (one of: "Bernstein", "Bilandzija", "Ervin", "Geckle", "Laska", "Lin", "Luthey", "Makhoul", "Mascheroni", "Oshiro", "Staley", "Turnipseed", "Velarde", "Wild" - match partial names like "ms masch" -> "Mascheroni", "mr biland" -> "Bilandzija", "turnip" -> "Turnipseed"),
  "first_name_contains": array of lowercase strings/letters that must appear in the student's first name (e.g. ["e"]),
  "last_name_contains": array of lowercase strings/letters that must appear in the student's last name (e.g. ["e"]),
  "first_name_starts_with": lowercase string or null,
  "first_name_ends_with": lowercase string or null,
  "last_name_starts_with": lowercase string or null,
  "last_name_ends_with": lowercase string or null,
  "id_contains": string of digits or null,
  "hyphenated_last": boolean,
  "double_letters_in_last": boolean,
  "explanation": "A crisp 1-sentence summary of the filter criteria applied."
}
Important rules:
1. If the user says "an e in his first and last name", put ["e"] in BOTH "first_name_contains" AND "last_name_contains".
2. If the user mentions "boy", "boys", "male", "his", "him", "guy", set "gender": "boy".
3. If the user mentions "girl", "girls", "female", "her", "she", set "gender": "girl".
4. If the user mentions "ms masch" or "masch", set "homeroom_teacher": "Mascheroni".
5. Output ONLY raw JSON, no markdown fences.`;

function blankSpan(text, start, end) {
  return text.slice(0, start) + " ".repeat(end - start) + text.slice(end);
}

export function heuristicSpec(prompt) {
  const p = (prompt || "").toLowerCase();
  let scrubbed = p;
  const grab = (m, group = 0) => {
    if (!m) return;
    const s = m.indices && m.indices[group] ? m.indices[group][0] : m.index;
    const e = m.indices && m.indices[group] ? m.indices[group][1] : m.index + m[0].length;
    if (s >= 0) scrubbed = blankSpan(scrubbed, s, e);
  };

  const spec = {
    gender: null,
    homeroom_teacher: null,
    first_name_contains: [],
    last_name_contains: [],
    first_name_starts_with: null,
    first_name_ends_with: null,
    last_name_starts_with: null,
    last_name_ends_with: null,
    id_contains: null,
    hyphenated_last: false,
    hyphenated_first: false,
    double_letters_in_last: false,
    double_letters_in_first: false,
  };

  for (const t of CLIENT_TEACHERS) {
    const tl = t.toLowerCase();
    const prefs = [tl];
    if (tl.length >= 5) for (let k = 4; k < tl.length; k++) prefs.push(tl.slice(0, k));
    const hit = prefs.some((pref) => new RegExp(`\\b${pref}('s)?\\b`).test(p));
    if (hit) { spec.homeroom_teacher = t; break; }
  }

  const pushBoth = (letter) => {
    if (letter && !spec.first_name_contains.includes(letter)) spec.first_name_contains.push(letter);
    if (letter && !spec.last_name_contains.includes(letter)) spec.last_name_contains.push(letter);
  };

  let m = p.match(/(?:letter\s+|an?\s+)['"]?([a-z])['"]?\s+in\s+(?:both\s+)?(?:his|her|their|hisn|the)?\s*first\s+(?:name\s+)?and\s+last\s+name/d);
  if (m) { pushBoth(m[1]); grab(m, 1); }
  m = p.match(/first\s+(?:name\s+)?and\s+last\s+name\s+(?:both\s+)?(?:with|containing|contain|have|has|having)\s+(?:the\s+letter\s+|an?\s+)?['"]?([a-z])['"]?\b/d);
  if (m) { pushBoth(m[1]); grab(m, 1); }
  m = p.match(/(?:with|containing|contain|having|that\s+have|that\s+contain)\s+(?:the\s+letter\s+|an?\s+)?['"]?([a-z])['"]?\s+in\s+(?:both\s+)?(?:their|his|her|the)?\s*first\s+(?:name\s+)?and\s+last\s+name/d);
  if (m) { pushBoth(m[1]); grab(m, 1); }

  for (const r of p.matchAll(/(?:letter\s+|an?\s+)['"]?([a-z])['"]?\s+in\s+(?:his|her|their|the)?\s*first\s+name/dg)) {
    if (!spec.first_name_contains.includes(r[1])) spec.first_name_contains.push(r[1]);
    grab(r, 1);
  }
  for (const r of p.matchAll(/(?:letter\s+|an?\s+)['"]?([a-z])['"]?\s+in\s+(?:his|her|their|the)?\s*last\s+name/dg)) {
    if (!spec.last_name_contains.includes(r[1])) spec.last_name_contains.push(r[1]);
    grab(r, 1);
  }

  m = p.match(/first\s+name\s+(?:starts|starting|begins|beginning)\s+with\s+(?:the\s+letter\s+|an?\s+)?['"]?([a-z]+)['"]?/d);
  if (m) { spec.first_name_starts_with = m[1]; grab(m, 1); }
  m = p.match(/last\s+name\s+(?:starts|starting|begins|beginning)\s+with\s+(?:the\s+letter\s+|an?\s+)?['"]?([a-z]+)['"]?/d);
  if (m) { spec.last_name_starts_with = m[1]; grab(m, 1); }
  if (!spec.first_name_starts_with && !spec.last_name_starts_with) {
    m = p.match(/\bname\s+(?:starts|starting|begins|beginning)\s+with\s+(?:the\s+letter\s+|an?\s+)?['"]?([a-z])['"]?\b/d);
    if (m) { spec.first_name_starts_with = m[1]; grab(m, 1); }
  }
  m = p.match(/first\s+name\s+(?:ends|ending)\s+with\s+(?:the\s+letter\s+|an?\s+)?['"]?([a-z]+)['"]?/d);
  if (m) { spec.first_name_ends_with = m[1]; grab(m, 1); }
  m = p.match(/last\s+name\s+(?:ends|ending)\s+with\s+(?:the\s+letter\s+|an?\s+)?['"]?([a-z]+)['"]?/d);
  if (m) { spec.last_name_ends_with = m[1]; grab(m, 1); }

  if (p.includes("hyphen")) {
    if (p.includes("first") && !p.includes("last")) spec.hyphenated_first = true;
    else spec.hyphenated_last = true;
  }
  if (p.includes("double letter") || p.includes("double-letter")) {
    if (p.includes("first") && !p.includes("last")) spec.double_letters_in_first = true;
    else spec.double_letters_in_last = true;
  }
  m = p.match(/\bid\s+(?:number\s+)?(?:containing|with|starts\s+with|ending\s+in|is)?\s*([0-9]{2,8})\b/d);
  if (m) { spec.id_contains = m[1]; grab(m, 1); }

  if (/\b(boy|boys|male|males|guy|guys|he|his|him|son)\b/.test(scrubbed)) spec.gender = "boy";
  else if (/\b(girl|girls|female|females|gal|gals|she|her|hers|daughter)\b/.test(scrubbed)) spec.gender = "girl";

  return spec;
}

function normList(val) {
  if (!val) return [];
  if (typeof val === "string") return val.trim() ? [val.toLowerCase().trim()] : [];
  if (Array.isArray(val)) return val.map((x) => String(x).toLowerCase().trim()).filter(Boolean);
  return [];
}

export function mergeSpecs(llm, heur) {
  const merged = {};
  const lg = ((llm && llm.gender) || "").toLowerCase().trim();
  const llmGender = ["male", "boy", "boys", "m"].includes(lg) ? "boy"
    : ["female", "girl", "girls", "f"].includes(lg) ? "girl" : null;
  merged.gender = (heur && heur.gender) || llmGender || null;

  const aliasHit = (alias) => {
    if (!alias) return null;
    let c = String(alias).replace(/^(mr\.?|mrs\.?|ms\.?|miss|dr\.?|prof\.?|teacher|homeroom)\s+/i, "");
    c = c.replace(/'s(\s+homeroom|\s+class)?$/i, "").trim().toLowerCase();
    if (!c) return null;
    return CLIENT_TEACHERS.find((t) => {
      const tl = t.toLowerCase();
      return c === tl || tl.startsWith(c) || tl.includes(c);
    }) || null;
  };
  merged.homeroom_teacher = (heur && heur.homeroom_teacher) || aliasHit(llm && llm.homeroom_teacher) || null;

  let fc = [...new Set([...((heur && heur.first_name_contains) || []), ...normList(llm && llm.first_name_contains)])];
  let lc = [...new Set([...((heur && heur.last_name_contains) || []), ...normList(llm && llm.last_name_contains)])];
  if (heur && heur.first_name_contains.length && heur.last_name_contains.length) {
    fc = heur.first_name_contains;
    lc = heur.last_name_contains;
  }
  merged.first_name_contains = fc;
  merged.last_name_contains = lc;

  for (const key of ["first_name_starts_with", "first_name_ends_with", "last_name_starts_with", "last_name_ends_with", "id_contains"]) {
    const v = (heur && heur[key]) || (llm && llm[key]);
    merged[key] = v ? String(v).toLowerCase().trim() : null;
  }
  merged.hyphenated_last = Boolean((heur && heur.hyphenated_last) || (llm && llm.hyphenated_last));
  merged.hyphenated_first = Boolean(heur && heur.hyphenated_first);
  merged.double_letters_in_last = Boolean((heur && heur.double_letters_in_last) || (llm && llm.double_letters_in_last));
  merged.double_letters_in_first = Boolean(heur && heur.double_letters_in_first);
  merged.explanation = (llm && llm.explanation) || "Filter compiled on-device against 336 student roster records.";
  return merged;
}

export function applySpec(students, spec) {
  const out = [];
  for (const s of students) {
    const first = (s.first_name || "").toLowerCase();
    const last = (s.last_name || "").toLowerCase();
    const teacher = (s.homeroom_teacher || "").toLowerCase();
    const sid = String(s.id || "").toLowerCase();
    if (spec.gender === "boy" && !s.is_boy) continue;
    if (spec.gender === "girl" && !s.is_girl) continue;
    if (spec.homeroom_teacher && teacher !== spec.homeroom_teacher.toLowerCase()) continue;
    if ((spec.first_name_contains || []).some((sub) => !first.includes(sub))) continue;
    if ((spec.last_name_contains || []).some((sub) => !last.includes(sub))) continue;
    if (spec.first_name_starts_with && !first.startsWith(spec.first_name_starts_with)) continue;
    if (spec.first_name_ends_with && !first.endsWith(spec.first_name_ends_with)) continue;
    if (spec.last_name_starts_with && !last.startsWith(spec.last_name_starts_with)) continue;
    if (spec.last_name_ends_with && !last.endsWith(spec.last_name_ends_with)) continue;
    if (spec.id_contains && !sid.includes(spec.id_contains)) continue;
    if (spec.hyphenated_last && !last.includes("-")) continue;
    if (spec.hyphenated_first && !first.includes("-")) continue;
    if (spec.double_letters_in_last && !/([a-z])\1/.test(last)) continue;
    if (spec.double_letters_in_first && !/([a-z])\1/.test(first)) continue;
    const reasons = [];
    if (spec.gender) reasons.push(`Gender: ${spec.gender.toUpperCase()}`);
    if (spec.homeroom_teacher) reasons.push(`Homeroom: ${s.homeroom_teacher}`);
    for (const sub of spec.first_name_contains || []) reasons.push(`First has '${sub.toUpperCase()}'`);
    for (const sub of spec.last_name_contains || []) reasons.push(`Last has '${sub.toUpperCase()}'`);
    if (spec.first_name_starts_with) reasons.push(`First starts '${spec.first_name_starts_with.toUpperCase()}'`);
    if (spec.last_name_starts_with) reasons.push(`Last starts '${spec.last_name_starts_with.toUpperCase()}'`);
    out.push({ ...s, match_reasons: reasons });
  }
  return out;
}

export function hasAnyConstraint(spec) {
  return Boolean(
    spec.gender || spec.homeroom_teacher ||
    (spec.first_name_contains || []).length || (spec.last_name_contains || []).length ||
    spec.first_name_starts_with || spec.first_name_ends_with ||
    spec.last_name_starts_with || spec.last_name_ends_with || spec.id_contains ||
    spec.hyphenated_last || spec.hyphenated_first ||
    spec.double_letters_in_last || spec.double_letters_in_first
  );
}

/* ---------- In-browser LLM assist (downloaded once, cached by the browser) ---------- */

let enginePromise = null;

export function webGpuAvailable() {
  return Boolean(navigator.gpu);
}

export function getBrowserEngine(onProgress) {
  if (!webGpuAvailable()) {
    return Promise.reject(new Error("WebGPU is unavailable in this browser or device."));
  }
  if (!enginePromise) {
    enginePromise = (async () => {
      const mod = await import(/* @vite-ignore */ WEBLLM_CDN);
      const engine = await mod.CreateMLCEngine(BROWSER_MODEL_ID, {
        initProgressCallback: (p) => {
          if (onProgress) onProgress(p);
        },
      });
      return engine;
    })().catch((err) => {
      enginePromise = null;
      throw err;
    });
  } else if (onProgress) {
    enginePromise.then(() => onProgress({ progress: 1, text: "Model ready from browser cache." })).catch(() => {});
  }
  return enginePromise;
}

function salvageJson(content) {
  if (!content) return {};
  const tick = String.fromCharCode(96).repeat(3);
  const text = content.split("\n").filter((ln) => !ln.trim().startsWith(tick)).join("\n").trim();
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object") return parsed;
  } catch { /* fall through to brace search */ }
  const brace = text.match(/\{[\s\S]*\}/);
  if (brace) {
    try {
      const parsed = JSON.parse(brace[0]);
      if (parsed && typeof parsed === "object") return parsed;
    } catch { /* ignore */ }
  }
  return {};
}

export async function queryBrowserLLM(prompt, onProgress) {
  const engine = await getBrowserEngine(onProgress);
  const res = await engine.chat.completions.create({
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ],
    temperature: 0.0,
    response_format: { type: "json_object" },
  });
  return salvageJson(res.choices[0].message.content || "");
}

/* Full on-device flow: deterministic spec, optional browser-LLM refine,
   same zero-hit guard as the server. */
export async function clientFilterFlow(students, prompt, { useBrowserAI, log } = {}) {
  const t0 = performance.now();
  const heur = heuristicSpec(prompt);
  let llm = {};
  let llmOk = false;
  let modelUsed = "deterministic";
  let note = "";

  if (useBrowserAI) {
    try {
      if (log) log("Downloading browser AI model on first run (about 700 MB, cached after)...");
      llm = await queryBrowserLLM(prompt, (p) => {
        if (log && p && typeof p.progress === "number") {
          const pct = Math.round(p.progress * 100);
          log(`Browser model ${p.text || ""} (${pct} percent)`);
        }
      });
      llmOk = Boolean(llm && Object.keys(llm).length);
      modelUsed = BROWSER_MODEL_ID.split("-Instruct")[0].toLowerCase();
      if (log) log(`Browser model compiled predicate: ${JSON.stringify(llm)}`, true);
    } catch (err) {
      note = `Browser AI unavailable (${err.message}); deterministic parser answered.`;
      if (log) log(note);
    }
  }

  const heurMerged = mergeSpecs({}, heur);
  const heurHits = applySpec(students, heurMerged);
  const merged = mergeSpecs(llm, heur);
  let hits = applySpec(students, merged);
  let llmKept = Boolean(llmOk && Object.keys(llm).length);
  if (hits.length === 0 && heurHits.length > 0) {
    hits = heurHits;
    llmKept = false;
    note = note || "Browser extras discarded after zero-hit validation; deterministic answer kept.";
    if (log) log(note);
    return { students: hits, spec: heurMerged, engine: "deterministic", model: modelUsed, llmOk, note, elapsedMs: Math.round(performance.now() - t0) };
  }
  return {
    students: hits,
    spec: merged,
    engine: llmOk && llmKept ? "browser-ai+deterministic" : "deterministic",
    model: modelUsed,
    llmOk,
    note,
    elapsedMs: Math.round(performance.now() - t0),
  };
}
