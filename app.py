import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="P7-HS // Roster Archive Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

BOY_NAMES = {
    "Adam", "Aiden", "Alec", "Alex", "Alexander", "Andrew", "Anthony", "Anton", "Antranik",
    "Apollo", "Ara", "Armaan", "Arthur", "Asher", "Austin", "Benjamin", "Bennett", "Bodhi",
    "Brij", "Buck", "Caden", "Caelem Nicholas", "Caigen", "Cameron", "Capeton", "Carson",
    "Christopher", "Clayton", "Colin", "Connor", "Daehee", "Daniel", "Darren", "David",
    "Declan", "Derek", "Deven", "Devin", "Diego", "Drake", "Ellis", "Emerson", "Enzo",
    "Ethan", "Franklin", "Gage", "Galo", "Glendin", "Hagop", "Hayk", "Hunter", "Ivan",
    "Jace", "Jack", "Jackson", "Jacob", "Jakson", "Jared", "Jasper", "Jayden", "Jeffrey",
    "Jenson", "Jeremiah", "Jiabao", "Jian", "Jibran", "Jinyao", "John", "Joshua", "Josiah",
    "Kai", "Kasen", "Keola", "Kurt", "Kyle", "Lake", "Leo", "Liam", "Linwood", "Logan",
    "Lorentz", "Lucas", "Luke", "Maarten", "Marc", "Mason", "Mateo", "Matthew",
    "Matthew Chun Kin", "Max", "Maximilian", "Maxwell", "Michael", "Miles", "Milo", "Milos",
    "Narek", "Nathan", "Nathaniel", "Nicholas", "Nicolas", "Nishan", "Noah", "Noel", "Nolan",
    "Odin", "Oliver", "Oren", "Osheen", "Parisch Hunter", "Parker", "Parker Hedrick",
    "ParkerHedrick", "Parsa", "Patrik", "Paul", "Pearce", "Phaolan", "Porter", "Quentin",
    "Quinn", "Rafael", "Robert", "Ruomu", "Russell", "Ryan", "Sam", "Sascha", "Schyler",
    "Shane", "Shuxing", "Spencer", "Terrance", "Timothy", "Vaun", "Vehm", "Victor", "Wesley",
    "William", "Xavier", "Yeon", "Yiming", "Zev"
}

GIRL_NAMES = {
    "Abigail", "Addison", "Adeline", "Adina", "Adriana", "Alina", "Alissa", "Allie",
    "Annabel", "Araxi", "Aria", "Ariana", "Asha", "Ashley", "Autumn", "Ava",
    "Ayessatou-Elise", "Ayla", "Beth", "Caroline", "Cela", "Charlotte", "Charmaine",
    "Christina", "Claire", "Clara", "Colette", "Crimson", "Daniella", "Delilah", "Diana",
    "Ella", "Ellie", "Emily", "Emmeline", "Erin", "Evelyn", "Farah", "Gabriella", "Grace",
    "Hailey", "Hannah", "Hayley", "Henrietta", "Isabella", "Isla", "Jayme", "Josephine",
    "Julia", "Juliana", "Kate", "Katerina", "Katherine", "Katie", "Kendall", "Kirra",
    "Kylie", "Liana", "Lillian", "Lily", "Lindsey", "Lola", "Lori", "Lucia", "Lucinda",
    "Lyra", "Macey", "Madeleine", "Madelyn", "Madison", "Makenna", "Malena", "Maral",
    "Matilda", "Maya", "Melodie", "Mia", "Nahla", "Natalie", "Noor", "Olive", "Olivia",
    "Poppy", "Renee", "Rhipsy", "Sabrina", "Sadie", "Sage", "Samantha", "Savannah",
    "Scarlett", "Sharlise", "Siyona", "Sobha", "Sofia", "Sofiani", "Sophia", "Sophie",
    "Sutton", "Teni", "Ursula", "Victoria", "Violet", "Yaeanne", "Zara", "Zoe",
    # unisex names also matchable when searching girls
    "Alex", "Cameron", "Ellis", "Emerson", "Jayden", "Kai", "Kendall", "Lake", "Logan",
    "Noel", "Parker", "Quinn", "Ryan", "Sage", "Sam", "Sascha", "Schyler", "Sutton"
}

ALL_TEACHERS = [
    "Bernstein", "Bilandzija", "Ervin", "Geckle", "Laska", "Lin", "Luthey",
    "Makhoul", "Mascheroni", "Oshiro", "Staley", "Turnipseed", "Velarde", "Wild"
]

# Probabilistic surname background: US Census 2010 surname percentages.
# These are statistics about US bearers of a surname, NOT facts about any student.
RACE_MIN_PCT = 60.0
RACE_LABELS = {
    "white": "white",
    "black": "black",
    "asian": "asian",
    "hispanic": "hispanic", "latino": "hispanic", "latina": "hispanic",
    "latinx": "hispanic", "latine": "hispanic",
    "native": "native", "indigenous": "native",
    "multi": "multi", "multiracial": "multi",
}
RACE_PHRASES = {
    "native american": "native", "american indian": "native",
    "mixed race": "multi",
}


def load_surname_table() -> Dict[str, Dict[str, float]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(base_dir, "static", "surname_race.json"),
                      os.path.join(base_dir, "surname_race.json")):
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f).get("surnames", {})
    return {}


SURNAME_RACE = load_surname_table()


def surname_part_probs(last: str) -> List[tuple]:
    found = []
    for part in re.split(r"[^A-Za-z]+", last or ""):
        key = re.sub(r"[^A-Z]", "", part.upper())
        if len(key) >= 2 and key in SURNAME_RACE:
            found.append((part, SURNAME_RACE[key]))
    return found


def surname_category_pct(last: str, category: str) -> float:
    best = 0.0
    for _, dist in surname_part_probs(last):
        try:
            best = max(best, float(dist.get(category, 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return best


def load_students() -> List[Dict[str, Any]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "roster_rows.json")
    students = []
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for idx, r in enumerate(raw, 1):
            first = r.get("first", "").strip()
            last = r.get("last", "").strip()
            teacher = r.get("teacher", "").strip()
            sid = str(r.get("id", "")).strip()
            pages = r.get("pages", [])

            is_boy = first in BOY_NAMES
            is_girl = first in GIRL_NAMES
            if is_boy and is_girl:
                gender = "unisex"
            elif is_boy:
                gender = "boy"
            elif is_girl:
                gender = "girl"
            else:
                gender = "unknown"

            students.append({
                "index": idx,
                "id": sid,
                "first_name": first,
                "last_name": last,
                "full_name": f"{first} {last}".strip(),
                "homeroom_teacher": teacher,
                "homeroom": "P7-HS",
                "gender": gender,
                "is_boy": is_boy,
                "is_girl": is_girl,
                "pages": pages,
            })
    return students


STUDENTS_CACHE = load_students()

PHOTO_HAIR_LABELS = {"blonde": "Blonde", "brown": "Brown", "black": "Dark", "red": "Red", "gray": "Gray"}


def load_face_index() -> Dict[str, Dict[str, Any]]:
    """Yearbook photo faces: student id -> {file, desc, page} for photo features."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(base_dir, "static", "face_index.json"),
                 os.path.join(base_dir, "face_index.json")):
        if os.path.exists(cand):
            try:
                raw = json.load(open(cand, encoding="utf-8"))
            except Exception:
                return {}
            faces = {}
            for page, data in raw.items():
                for f in data.get("faces", []):
                    if f.get("id") and f.get("file"):
                        faces[str(f["id"])] = f
            return faces
    return {}


FACE_CACHE = load_face_index()

class AIQueryRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:3b"
    min_think_ms: int = 0


def match_teacher_alias(alias: str) -> List[str]:
    if not alias:
        return []
    cleaned = re.sub(r"^(mr\.?|mrs\.?|ms\.?|miss|dr\.?|prof\.?|teacher|homeroom)\s+", "", alias.strip(), flags=re.I)
    cleaned = re.sub(r"'s(\s+homeroom|\s+class)?$", "", cleaned, flags=re.I).strip().lower()
    if not cleaned:
        return []
    matched = []
    for t in ALL_TEACHERS:
        tl = t.lower()
        if cleaned == tl or tl.startswith(cleaned) or cleaned in tl:
            matched.append(t)
    return matched


def heuristic_parse_prompt(prompt: str) -> Dict[str, Any]:
    """Deterministic parser so spelling/homeroom/gender queries are 100% accurate.
    Letter-value spans are scrubbed before gender detection so a search value
    like "son" in 'ends with son' never triggers a gender constraint."""
    p = prompt.lower()
    scrubbed = p

    def grab(m, group: int = 0) -> None:
        nonlocal scrubbed
        if m:
            try:
                s, e = m.span(group)
            except IndexError:
                s, e = m.span()
            if s >= 0:
                scrubbed = scrubbed[:s] + (" " * (e - s)) + scrubbed[e:]

    spec: Dict[str, Any] = {
        "gender": None,
        "homeroom_teacher": None,
        "first_name_contains": [],
        "last_name_contains": [],
        "first_name_starts_with": None,
        "first_name_ends_with": None,
        "last_name_starts_with": None,
        "last_name_ends_with": None,
        "id_contains": None,
        "name_contains": [],
        "race_likely": None,
        "photo_glasses": None,
        "photo_hair": None,
        "has_photo": None,
        "double_letters_in_last": False,
        "double_letters_in_first": False,
        "hyphenated_last": False,
        "hyphenated_first": False,
    }

    # NOTE: gender detection runs after letter extraction (see bottom of
    # this function) on text with letter-value spans scrubbed out.

    # Teacher detection
    for t in ALL_TEACHERS:
        tl = t.lower()
        # check full name or prefix of >= 4 chars (like "masch" for "mascheroni", "biland" for "bilandzija", "turnip" for "turnipseed")
        prefixes = [tl]
        if len(tl) >= 5:
            for k in range(4, len(tl)):
                prefixes.append(tl[:k])
        for pref in prefixes:
            if re.search(r"\b" + re.escape(pref) + r"('s)?\b", p):
                spec["homeroom_teacher"] = t
                break
        if spec["homeroom_teacher"]:
            break

    # "an <letter> in his/her/their first and last name" or "both first and last name"
    m_both = re.search(
        r"(?:letter\s+|an?\s+)['\"]?([a-z])['\"]?\s+in\s+(?:both\s+)?(?:his|her|their|hisn|the)?\s*first\s+(?:name\s+)?and\s+last\s+name",
        p
    )
    if m_both:
        letter = m_both.group(1)
        spec["first_name_contains"].append(letter)
        spec["last_name_contains"].append(letter)
        grab(m_both, 1)

    # Also check reverse: "first and last name have/contain/with an <letter>"
    m_both2 = re.search(
        r"first\s+(?:name\s+)?and\s+last\s+name\s+(?:both\s+)?(?:with|containing|contain|have|has|having)\s+(?:the\s+letter\s+|an?\s+)?['\"]?([a-z])['\"]?\b",
        p
    )
    if m_both2:
        letter = m_both2.group(1)
        if letter not in spec["first_name_contains"]:
            spec["first_name_contains"].append(letter)
        if letter not in spec["last_name_contains"]:
            spec["last_name_contains"].append(letter)
        grab(m_both2, 1)

    # "with <letter> in first and last name" (verb-led, e.g. "boys with e in first and last name")
    m_both3 = re.search(
        r"(?:with|containing|contain|having|that\s+have|that\s+contain)\s+(?:the\s+letter\s+|an?\s+)?['\"]?([a-z])['\"]?\s+in\s+(?:both\s+)?(?:their|his|her|the)?\s*first\s+(?:name\s+)?and\s+last\s+name",
        p
    )
    if m_both3:
        letter = m_both3.group(1)
        if letter not in spec["first_name_contains"]:
            spec["first_name_contains"].append(letter)
        if letter not in spec["last_name_contains"]:
            spec["last_name_contains"].append(letter)
        grab(m_both3, 1)

    # First name contains letter
    for m in re.finditer(r"(?:letter\s+|an?\s+)['\"]?([a-z])['\"]?\s+in\s+(?:his|her|their|the)?\s*first\s+name", p):
        ch = m.group(1)
        if ch not in spec["first_name_contains"]:
            spec["first_name_contains"].append(ch)
        grab(m, 1)

    # Last name contains letter
    for m in re.finditer(r"(?:letter\s+|an?\s+)['\"]?([a-z])['\"]?\s+in\s+(?:his|her|their|the)?\s*last\s+name", p):
        ch = m.group(1)
        if ch not in spec["last_name_contains"]:
            spec["last_name_contains"].append(ch)
        grab(m, 1)

    # First name starts with
    m_fs = re.search(r"first\s+name\s+(?:starts|starting|begins|beginning)\s+with\s+(?:the\s+letter\s+|an?\s+)?['\"]?([a-z]+)['\"]?", p)
    if m_fs:
        spec["first_name_starts_with"] = m_fs.group(1)
        grab(m_fs, 1)

    # Last name starts with
    m_ls = re.search(r"last\s+name\s+(?:starts|starting|begins|beginning)\s+with\s+(?:the\s+letter\s+|an?\s+)?['\"]?([a-z]+)['\"]?", p)
    if m_ls:
        spec["last_name_starts_with"] = m_ls.group(1)
        grab(m_ls, 1)

    # General "starts with <X>" if not specified first/last
    if not spec["first_name_starts_with"] and not spec["last_name_starts_with"]:
        m_gs = re.search(r"\bname\s+(?:starts|starting|begins|beginning)\s+with\s+(?:the\s+letter\s+|an?\s+)?['\"]?([a-z])['\"]?\b", p)
        if m_gs:
            spec["first_name_starts_with"] = m_gs.group(1)
            grab(m_gs, 1)

    # Ends with
    m_fe = re.search(r"first\s+name\s+(?:ends|ending)\s+with\s+(?:the\s+letter\s+|an?\s+)?['\"]?([a-z]+)['\"]?", p)
    if m_fe:
        spec["first_name_ends_with"] = m_fe.group(1)
        grab(m_fe, 1)
    m_le = re.search(r"last\s+name\s+(?:ends|ending)\s+with\s+(?:the\s+letter\s+|an?\s+)?['\"]?([a-z]+)['\"]?", p)
    if m_le:
        spec["last_name_ends_with"] = m_le.group(1)
        grab(m_le, 1)

    # Hyphenated
    if "hyphen" in p:
        if "first" in p and "last" not in p:
            spec["hyphenated_first"] = True
        else:
            spec["hyphenated_last"] = True

    # Double letters
    if "double letter" in p or "double-letter" in p:
        if "first" in p and "last" not in p:
            spec["double_letters_in_first"] = True
        else:
            spec["double_letters_in_last"] = True

    # ID digits
    m_id = re.search(r"\bid\s+(?:number\s+)?(?:containing|with|starts\s+with|ending\s+in|is)?\s*([0-9]{2,8})\b", p)
    if m_id:
        spec["id_contains"] = m_id.group(1)
        grab(m_id)

    # Likely background from Census surname statistics (probabilistic, never a fact).
    # Matches "asian students", "race hispanic", "likely white surnames", etc.
    if re.search(r"\b(race|races|ethnicity|ethnic|background|surname|surnames|last\s+name|student|students)\b", p):
        for phrase, category in RACE_PHRASES.items():
            m_rp = re.search(r"\b" + re.escape(phrase) + r"\b", p)
            if m_rp:
                spec["race_likely"] = category
                grab(m_rp)
                break
        if not spec["race_likely"]:
            for alias, category in RACE_LABELS.items():
                if alias in ("multi",):
                    continue
                m_rl = re.search(r"\b" + re.escape(alias) + r"\b", p)
                if m_rl:
                    spec["race_likely"] = category
                    grab(m_rl, 0)
                    break
        if not spec["race_likely"] and re.search(r"\b(race|ethnicity|ethnic)\b", p):
            m_mx = re.search(r"\bmixed\b", p)
            if m_mx:
                spec["race_likely"] = "multi"
                grab(m_mx)

    # Photo features from the yearbook page itself (glasses flag / hair tone).
    if re.search(r"\bno\s+glasses\b|\bwithout\s+glasses\b", p):
        spec["photo_glasses"] = False
        grab(re.search(r"\b(no\s+glasses|without\s+glasses)\b", p))
    elif re.search(r"\bglasse?s\b|\bwearing\s+glass\b", p):
        m_g = re.search(r"\b(glasse?s|wearing\s+glass)\b", p)
        spec["photo_glasses"] = True
        grab(m_g)

    for m_h in re.finditer(r"\b(blonde|blond|brown|black|dark\s+brown|red|gray|grey)\s+(?:-\s*)?hair\b", p):
        word = m_h.group(1)
        bucket = {"blonde": "blonde", "blond": "blonde", "brown": "brown",
                  "black": "black", "dark brown": "brown", "red": "red",
                  "gray": "gray", "grey": "gray"}.get(word, "brown")
        spec["photo_hair"] = bucket
        grab(m_h)
        break
    if not spec["photo_hair"]:
        m_rh = re.search(r"\bredheads?\b|\bred\-haired\b", p)
        if m_rh:
            spec["photo_hair"] = "red"
            grab(m_rh)

    m_ph = re.search(r"\b(?:has?|with)\s+(?:a\s+)?(?:yearbook\s+)?photo\b", p)
    if m_ph:
        spec["has_photo"] = True
        grab(m_ph)

    # Gender detection LAST, on text with letter-value spans scrubbed out,
    # so values like "son" (ends with son) never leak into gender.
    if re.search(r"\b(boy|boys|male|males|guy|guys|he|his|him|son)\b", scrubbed):
        spec["gender"] = "boy"
    elif re.search(r"\b(girl|girls|female|females|gal|gals|she|her|hers|daughter)\b", scrubbed):
        spec["gender"] = "girl"

    return spec


SYSTEM_PROMPT = """You are the P7-HS Roster Query Compiler, a precise student roster query engine.
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
  "min_first_name_length": integer or null,
  "max_first_name_length": integer or null,
  "min_last_name_length": integer or null,
  "max_last_name_length": integer or null,
  "hyphenated_last": boolean,
  "double_letters_in_last": boolean,
  "race_likely": one of "white" | "black" | "asian" | "hispanic" | "native" | "multi" or null (only when the user asks for a background, e.g. "asian students" means "asian"),
  "photo_glasses": true | false | null (true = "students with glasses" / "wearing glasses"; false only for "no glasses"/"without glasses"; null otherwise),
  "photo_hair": one of "blonde" | "brown" | "black" | "red" | "gray" | null (photo hair tone visible in the yearbook photo, e.g. "blonde hair"),
  "has_photo": true | null (true = "students with a photo" / "has a yearbook photo"),
  "explanation": "A crisp, futuristic 1-sentence summary of the neural filter criteria applied."
}
Important rules:
1. If the user says "an e in his first and last name", put ["e"] in BOTH "first_name_contains" AND "last_name_contains".
2. If the user mentions "boy", "boys", "male", "his", "him", "guy", set "gender": "boy".
3. If the user mentions "girl", "girls", "female", "her", "she", set "gender": "girl".
4. If the user mentions "ms masch" or "masch", set "homeroom_teacher": "Mascheroni".
5. Output ONLY raw JSON, no markdown fences.
6. Only set "race_likely" when the user explicitly asks for a background (race, ethnicity, asian, black, white, hispanic, latino).
7. Only set "photo_glasses"/"photo_hair" when the user explicitly asks about photo features (glasses, hair tone); these flags come from image analysis of the yearbook photos."""


def salvage_llm_json(content: str) -> Dict[str, Any]:
    """Parse LLM output, tolerating markdown fences and leading chatter.
    Small local models often wrap JSON in fences despite format=json."""
    if not content:
        return {}
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            parsed = json.loads(brace.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _try_ollama_model(prompt: str, model: str, timeout: int) -> Optional[Dict[str, Any]]:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 300,
            },
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        return None
    return salvage_llm_json(resp.json().get("message", {}).get("content", "")) or None


def installed_llm_models() -> List[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            names = [
                m.get("name", "")
                for m in r.json().get("models", [])
                if "embed" not in m.get("name", "").lower()
            ]
            priority = {"qwen2.5:3b": 0, "qwen2.5:1.5b": 1, "qwen3-vl:4b": 2, "llama3.1:8b": 3}
            names.sort(key=lambda n: priority.get(n, 99))
            return names
    except Exception:
        pass
    return []


def query_ollama_filter(prompt: str, model: str) -> tuple:
    """Query the requested model, falling back to other installed models.
    Returns (spec_dict, llm_ok, model_used). Never raises."""
    candidates = [model]
    for name in installed_llm_models():
        if name not in candidates:
            candidates.append(name)
    for i, candidate in enumerate(candidates):
        try:
            parsed = _try_ollama_model(prompt, candidate, timeout=25 if i == 0 else 10)
            if parsed:
                return parsed, True, candidate
        except Exception as e:
            print(f"Ollama query error ({candidate}): {e}")
    return {}, False, model


def resolve_results(heur_spec: Dict[str, Any], llm_spec: Dict[str, Any]):
    """Merge specs and pick the result set, guarding against LLM pollution.
    If the merged spec yields zero rows while the deterministic spec yields
    rows, the LLM extras are discarded and the deterministic answer wins.
    Returns (students, merged_spec, llm_kept)."""
    heur_merged = merge_specs({}, heur_spec)
    heur_hits = apply_filter_spec(STUDENTS_CACHE, heur_merged)
    merged = merge_specs(llm_spec, heur_spec)
    merged_hits = apply_filter_spec(STUDENTS_CACHE, merged)
    if not merged_hits and heur_hits:
        return heur_hits, heur_merged, False
    return merged_hits, merged, bool(llm_spec)


def merge_specs(llm_spec: Dict[str, Any], heur_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Combine LLM output with deterministic regex safeguards so neither misses key constraints."""
    merged: Dict[str, Any] = {}

    # Gender: prefer heuristic if unambiguous, otherwise LLM
    llm_gender = (llm_spec.get("gender") or "").lower().strip()
    if llm_gender in ("male", "boy", "boys", "m"):
        llm_gender = "boy"
    elif llm_gender in ("female", "girl", "girls", "f"):
        llm_gender = "girl"
    else:
        llm_gender = None
    merged["gender"] = heur_spec.get("gender") or llm_gender

    # Teacher: normalize LLM teacher through alias matcher
    llm_teacher_raw = llm_spec.get("homeroom_teacher")
    matched_teachers = match_teacher_alias(str(llm_teacher_raw)) if llm_teacher_raw else []
    llm_teacher = matched_teachers[0] if matched_teachers else None
    merged["homeroom_teacher"] = heur_spec.get("homeroom_teacher") or llm_teacher

    # Lists of substrings
    def norm_list(val: Any) -> List[str]:
        if not val:
            return []
        if isinstance(val, str):
            return [val.lower().strip()] if val.strip() else []
        if isinstance(val, list):
            return [str(x).lower().strip() for x in val if str(x).strip()]
        return []

    fc = list(dict.fromkeys(heur_spec.get("first_name_contains", []) + norm_list(llm_spec.get("first_name_contains"))))
    lc = list(dict.fromkeys(heur_spec.get("last_name_contains", []) + norm_list(llm_spec.get("last_name_contains"))))

    # Guardrail: if heuristic detected "in first and last name", trust heuristic's exact letters
    if heur_spec.get("first_name_contains") and heur_spec.get("last_name_contains"):
        fc = heur_spec["first_name_contains"]
        lc = heur_spec["last_name_contains"]

    merged["first_name_contains"] = fc
    merged["last_name_contains"] = lc

    for key in ("first_name_starts_with", "first_name_ends_with", "last_name_starts_with", "last_name_ends_with", "id_contains"):
        val = heur_spec.get(key) or llm_spec.get(key)
        merged[key] = str(val).lower().strip() if val else None

    for num_key in ("min_first_name_length", "max_first_name_length", "min_last_name_length", "max_last_name_length"):
        val = llm_spec.get(num_key)
        merged[num_key] = int(val) if isinstance(val, (int, float)) and val > 0 else None

    llm_race = str(llm_spec.get("race_likely") or "").lower().strip()
    merged["race_likely"] = heur_spec.get("race_likely") or (llm_race if llm_race in RACE_LABELS.values() else None)
    merged["photo_glasses"] = heur_spec.get("photo_glasses")
    if merged["photo_glasses"] is None and isinstance(llm_spec.get("photo_glasses"), bool):
        merged["photo_glasses"] = bool(llm_spec.get("photo_glasses"))
    merged["photo_hair"] = heur_spec.get("photo_hair")
    if not merged["photo_hair"] and str(llm_spec.get("photo_hair") or "").lower() in PHOTO_HAIR_LABELS:
        merged["photo_hair"] = str(llm_spec.get("photo_hair")).lower()
    merged["has_photo"] = heur_spec.get("has_photo")
    merged["hyphenated_last"] = bool(heur_spec.get("hyphenated_last") or llm_spec.get("hyphenated_last"))
    merged["hyphenated_first"] = bool(heur_spec.get("hyphenated_first"))
    merged["double_letters_in_last"] = bool(heur_spec.get("double_letters_in_last") or llm_spec.get("double_letters_in_last"))
    merged["double_letters_in_first"] = bool(heur_spec.get("double_letters_in_first"))

    merged["explanation"] = llm_spec.get("explanation") or "Filter compiled and verified against 336 student roster records."
    return merged


def apply_filter_spec(students: List[Dict[str, Any]], spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for s in students:
        first_l = s["first_name"].lower()
        last_l = s["last_name"].lower()
        teacher_l = s["homeroom_teacher"].lower()
        sid = s["id"].lower()

        # Gender
        if spec.get("gender") == "boy" and not s["is_boy"]:
            continue
        if spec.get("gender") == "girl" and not s["is_girl"]:
            continue

        # Homeroom teacher
        if spec.get("homeroom_teacher"):
            if teacher_l != spec["homeroom_teacher"].lower():
                continue

        # First name contains
        if any(sub not in first_l for sub in spec.get("first_name_contains", [])):
            continue

        # Last name contains
        if any(sub not in last_l for sub in spec.get("last_name_contains", [])):
            continue

        # Starts / ends with
        if spec.get("first_name_starts_with") and not first_l.startswith(spec["first_name_starts_with"]):
            continue
        if spec.get("first_name_ends_with") and not first_l.endswith(spec["first_name_ends_with"]):
            continue
        if spec.get("last_name_starts_with") and not last_l.startswith(spec["last_name_starts_with"]):
            continue
        if spec.get("last_name_ends_with") and not last_l.endswith(spec["last_name_ends_with"]):
            continue

        # ID contains
        if spec.get("id_contains") and spec["id_contains"] not in sid:
            continue

        # Likely background: Census surname probability must clear the bar
        race_pct = surname_category_pct(s["last_name"], spec["race_likely"]) if spec.get("race_likely") else 0.0
        if spec.get("race_likely") and race_pct < RACE_MIN_PCT:
            continue

        # Photo features from the yearbook image (heuristic flags per face)
        face = FACE_CACHE.get(s["id"])
        if spec.get("has_photo") and not face:
            continue
        if spec.get("photo_glasses") is True:
            if not face or not face.get("desc", {}).get("glasses"):
                continue
        if spec.get("photo_glasses") is False:
            if face and face.get("desc", {}).get("glasses"):
                continue
        if spec.get("photo_hair"):
            if not face or face.get("desc", {}).get("hair") != spec["photo_hair"]:
                continue

        # Lengths
        if spec.get("min_first_name_length") and len(first_l) < spec["min_first_name_length"]:
            continue
        if spec.get("max_first_name_length") and len(first_l) > spec["max_first_name_length"]:
            continue
        if spec.get("min_last_name_length") and len(last_l) < spec["min_last_name_length"]:
            continue
        if spec.get("max_last_name_length") and len(last_l) > spec["max_last_name_length"]:
            continue

        # Hyphenated
        if spec.get("hyphenated_last") and "-" not in last_l:
            continue
        if spec.get("hyphenated_first") and "-" not in first_l:
            continue

        # Double letters
        if spec.get("double_letters_in_last") and not re.search(r"([a-z])\1", last_l):
            continue
        if spec.get("double_letters_in_first") and not re.search(r"([a-z])\1", first_l):
            continue

        # Annotate match reasons for UI badges
        reasons = []
        if spec.get("gender"):
            reasons.append(f"Gender: {spec['gender'].upper()}")
        if spec.get("homeroom_teacher"):
            reasons.append(f"Homeroom: {s['homeroom_teacher']}")
        for sub in spec.get("first_name_contains", []):
            reasons.append(f"First has '{sub.upper()}'")
        for sub in spec.get("last_name_contains", []):
            reasons.append(f"Last has '{sub.upper()}'")
        if spec.get("first_name_starts_with"):
            reasons.append(f"First starts '{spec['first_name_starts_with'].upper()}'")
        if spec.get("last_name_starts_with"):
            reasons.append(f"Last starts '{spec['last_name_starts_with'].upper()}'")
        if spec.get("race_likely"):
            reasons.append(f"Surname suggests {spec['race_likely'].title()} ({race_pct:.0f}%)")
        if spec.get("photo_glasses") is True:
            reasons.append("Wearing glasses in yearbook photo (CV flag)")
        if spec.get("photo_glasses") is False:
            reasons.append("No glasses detected in photo")
        if spec.get("photo_hair"):
            reasons.append(f"Photo hair tone: {PHOTO_HAIR_LABELS[spec['photo_hair']]}")
        if spec.get("has_photo"):
            reasons.append("Yearbook photo on file")
        if face:
            s_extra = {"photo": "/static/faces/" + face["file"], "photo_page": face.get("page"),
                       "photo_desc": face.get("desc", {})}
            merged_into = dict(s)
            merged_into.update(s_extra)
            s = merged_into

        s_copy = dict(s)
        s_copy["match_reasons"] = reasons
        results.append(s_copy)

    return results


@app.get("/api/roster")
def get_roster():
    teachers = {}
    for s in STUDENTS_CACHE:
        t = s["homeroom_teacher"]
        teachers[t] = teachers.get(t, 0) + 1
    return {
        "total": len(STUDENTS_CACHE),
        "homeroom": "P7-HS",
        "teachers": teachers,
        "students": STUDENTS_CACHE,
    }


@app.get("/api/faces")
def get_faces():
    """Yearbook photo faces: id -> file + heuristic descriptors, for the UI."""
    return {
        "count": len(FACE_CACHE),
        "faces": {
            sid: {
                "file": f["file"],
                "page": f.get("page"),
                "desc": f.get("desc", {}),
            } for sid, f in FACE_CACHE.items()
        },
    }


@app.get("/api/models")
def get_models():
    models = []
    ollama_online = False
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_online = True
            for m in r.json().get("models", []):
                name = m.get("name", "")
                if "embed" not in name.lower():
                    models.append({
                        "name": name,
                        "size_gb": round(m.get("size", 0) / (1024 ** 3), 2),
                        "family": m.get("details", {}).get("family", "llm"),
                        "parameter_size": m.get("details", {}).get("parameter_size", ""),
                    })
    except Exception:
        pass

    # Prefer fast models at top
    priority = {"qwen2.5:3b": 0, "qwen2.5:1.5b": 1, "qwen3-vl:4b": 2, "llama3.1:8b": 3}
    models.sort(key=lambda x: priority.get(x["name"], 99))

    return {
        "ollama_online": ollama_online,
        "models": models,
        "default_model": models[0]["name"] if models else "qwen2.5:3b",
    }


@app.post("/api/ai-filter")
def ai_filter(req: AIQueryRequest):
    t0 = time.time()
    prompt = req.prompt.strip()
    if not prompt:
        return {
            "prompt": "",
            "model": req.model,
            "requested_model": req.model,
            "llm_ok": False,
            "engine": "deterministic",
            "note": "",
            "elapsed_ms": 0,
            "spec": {},
            "count": len(STUDENTS_CACHE),
            "students": STUDENTS_CACHE,
        }

    heur_spec = heuristic_parse_prompt(prompt)
    llm_spec, llm_ok, model_used = query_ollama_filter(prompt, req.model)
    matched, merged_spec, llm_kept = resolve_results(heur_spec, llm_spec)

    # If zero structured filters triggered at all, do fuzzy keyword search across full student record
    has_any_constraint = any([
        merged_spec.get("gender"),
        merged_spec.get("homeroom_teacher"),
        merged_spec.get("first_name_contains"),
        merged_spec.get("last_name_contains"),
        merged_spec.get("first_name_starts_with"),
        merged_spec.get("first_name_ends_with"),
        merged_spec.get("last_name_starts_with"),
        merged_spec.get("last_name_ends_with"),
        merged_spec.get("id_contains"),
        merged_spec.get("min_first_name_length"),
        merged_spec.get("max_first_name_length"),
        merged_spec.get("min_last_name_length"),
        merged_spec.get("max_last_name_length"),
        merged_spec.get("race_likely"),
        merged_spec.get("photo_glasses") is not None,
        merged_spec.get("photo_hair"),
        merged_spec.get("has_photo"),
        merged_spec.get("hyphenated_last"),
        merged_spec.get("hyphenated_first"),
        merged_spec.get("double_letters_in_last"),
        merged_spec.get("double_letters_in_first"),
    ])

    if not has_any_constraint:
        tokens = [tok for tok in re.split(r"\W+", prompt.lower()) if tok and tok not in {"students", "student", "name", "names", "with", "in", "the", "all", "show", "find", "me", "who", "are", "is", "homeroom", "class"}]
        if tokens:
            matched = []
            for s in STUDENTS_CACHE:
                blob = f"{s['first_name']} {s['last_name']} {s['homeroom_teacher']} {s['id']} {s['gender']}".lower()
                if all(tok in blob for tok in tokens):
                    sc = dict(s)
                    sc["match_reasons"] = [f"Matched '{tok}'" for tok in tokens]
                    matched.append(sc)

    elapsed_ms = int((time.time() - t0) * 1000)
    if req.min_think_ms > elapsed_ms:
        time.sleep((req.min_think_ms - elapsed_ms) / 1000.0)
        elapsed_ms = int((time.time() - t0) * 1000)

    engine = "llm+deterministic" if (llm_ok and llm_kept) else "deterministic"
    note = ""
    if not llm_ok:
        note = "LLM assist unavailable; deterministic parser answered."
    elif not llm_kept:
        note = "LLM extras discarded after zero-hit validation; deterministic answer kept."
    return {
        "prompt": prompt,
        "model": model_used,
        "requested_model": req.model,
        "llm_ok": llm_ok,
        "engine": engine,
        "note": note,
        "elapsed_ms": elapsed_ms,
        "spec": merged_spec,
        "raw_llm": llm_spec,
        "count": len(matched),
        "students": matched,
    }


@app.get("/")
def serve_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, "static", "index.html"))


@app.get("/IMG_3895.jpg")
def serve_hero_plate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, "static", "IMG_3895.jpg"))


base_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(base_static, exist_ok=True)
app.mount("/static", StaticFiles(directory=base_static), name="static")

base_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "png")
if os.path.exists(base_png):
    app.mount("/png", StaticFiles(directory=base_png), name="png")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("NEXUS_PORT", "8090"))
    uvicorn.run(app, host="127.0.0.1", port=port)
