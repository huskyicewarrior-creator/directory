"""Offline regression suite for the P7-HS roster filter (no Ollama needed).

Run:  python tests/test_filter_regression.py
Covers: deterministic parser (gender/teacher/letters), LLM-pollution guard,
JSON salvage, and exact end-to-end counts on the bundled 336-record roster.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    STUDENTS_CACHE,
    apply_filter_spec,
    heuristic_parse_prompt,
    merge_specs,
    resolve_results,
    salvage_llm_json,
)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" [{detail}]" if detail and not cond else ""))


def count_for(prompt, llm=None):
    heur = heuristic_parse_prompt(prompt)
    merged = merge_specs(llm or {}, heur)
    return len(apply_filter_spec(STUDENTS_CACHE, merged)), merged


# 1. Core preset: boys, Mascheroni, e in first and last (typo + possessive forms)
n, spec = count_for("a boys name with an e in hisn first and last name whos in ms masch's homeroom")
check("preset1 count == 3", n == 3, f"got {n}")
check("preset1 gender boy", spec.get("gender") == "boy")
check("preset1 teacher Mascheroni", spec.get("homeroom_teacher") == "Mascheroni")
check("preset1 letters e/e", spec.get("first_name_contains") == ["e"] and spec.get("last_name_contains") == ["e"])

# 2. Verb-led phrasing works without any LLM help
n2, spec2 = count_for("boys with e in first and last name in masch homeroom")
check("verb-led both-names count == 3", n2 == 3, f"got {n2}")
check("verb-led letters e/e", spec2.get("first_name_contains") == ["e"] and spec2.get("last_name_contains") == ["e"])

# 3. 'son' as a search value must not imply gender
_, spec3 = count_for("last name ends with son")
check("ends-with-son has no gender", spec3.get("gender") is None, f"got {spec3.get('gender')}")
check("ends-with-son lew == son", spec3.get("last_name_ends_with") == "son")
n3, _ = count_for("last name ends with son")
check("ends-with-son count == 5", n3 == 5, f"got {n3}")

# 4. Genuine possessive still signals gender
_, spec4 = count_for("an e in his first name")
check("his-signals-boy preserved", spec4.get("gender") == "boy", f"got {spec4.get('gender')}")

# 5. Remaining presets hold exact counts
checks_counts = [
    ("girls in Oshiro's homeroom with an a in their first name", 8),
    ("students with hyphenated last names", 4),
    ("boys in Turnipseed's homeroom", 15),
    ("students with double letters in their last name in Ervin's homeroom", 8),
    ("show me all girls", 152),
    ("first name starts with J", 25),
    ("id containing 2002", 56),
]
for prompt, expected in checks_counts:
    got, _ = count_for(prompt)
    check(f"count {expected} :: {prompt[:44]}", got == expected, f"got {got}")

# 6. LLM pollution guard: junk LLM extras that zero out results are discarded
heur = heuristic_parse_prompt("boys in Turnipseed's homeroom")
junk_llm = {"gender": None, "homeroom_teacher": None, "first_name_contains": ["zzz"],
            "last_name_contains": [], "explanation": "junk"}
students, used_spec, kept = resolve_results(heur, junk_llm)
check("polluting LLM discarded", kept is False and len(students) == 15, f"kept={kept} n={len(students)}")

# 7. Good LLM spec is kept
good_llm = {"gender": "boy", "homeroom_teacher": "Mascheroni",
            "first_name_contains": ["e"], "last_name_contains": ["e"], "explanation": "ok"}
students7, _, kept7 = resolve_results(heuristic_parse_prompt("boys with e in first and last name in masch homeroom"), good_llm)
check("good LLM spec kept", kept7 is True and len(students7) == 3, f"kept={kept7} n={len(students7)}")

# 8. Fenced JSON salvage (fence built without literal backticks)
fence = chr(96) * 3
fenced = fence + "json\n" + '{"gender": "girl", "first_name_contains": ["a"]}\n' + fence
parsed = salvage_llm_json(fenced)
check("fenced JSON salvaged", parsed.get("gender") == "girl", f"got {parsed}")
check("empty string salvages to {}", salvage_llm_json("") == {})
check("chatter prefix salvaged", salvage_llm_json('Sure! {"gender": null}') .get("gender") is None)

# 9. Likely-background search (Census surname statistics, 60%+ bar)
race_cases = [
    ("asian students", "asian"),
    ("hispanic students", "hispanic"),
    ("white students", "white"),
    ("black students", "black"),
    ("girls with likely asian surnames", "asian"),
    ("race hispanic in Oshiro homeroom", "hispanic"),
]
for prompt, category in race_cases:
    heur_r = heuristic_parse_prompt(prompt)
    check(f"race parsed :: {prompt[:40]}", heur_r.get("race_likely") == category, f"got {heur_r.get('race_likely')}")
    merged_r = merge_specs({}, heur_r)
    hits_r = apply_filter_spec(STUDENTS_CACHE, merged_r)
    check(f"race nonzero + badges :: {prompt[:40]}", len(hits_r) > 0 and all("Surname suggests" in " ".join(s.get("match_reasons", [])) for s in hits_r), f"n={len(hits_r)}")

n_asian, _ = count_for("asian students")
check("asian count == 85", n_asian == 85, f"got {n_asian}")
n_comb, spec_comb = count_for("race hispanic in Oshiro homeroom")
check("hispanic Oshiro count == 4", n_comb == 4, f"got {n_comb}")
check("race composes with teacher", spec_comb.get("homeroom_teacher") == "Oshiro")
_, spec_nb = count_for("boys in Turnipseed's homeroom")
check("no race leak into plain query", spec_nb.get("race_likely") is None)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed.")
sys.exit(1 if FAIL else 0)
