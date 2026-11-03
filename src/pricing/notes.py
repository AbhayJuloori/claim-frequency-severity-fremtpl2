"""Draft plain language segment notes with an LLM, then check every number against the table.

The LLM only writes prose. A deterministic checker extracts each number in the draft and
requires it to match a value in the segment table (after rounding) or a value derived
from it (percent forms, relativities). Drafts that cite any number not in the table are
rejected and retried; if the retry also fails, the note is marked unverified and the UI
shows the table instead of the prose.

Provider: a local Ollama model by default (no key, nothing leaves the machine). Set
LLM_PROVIDER=openrouter with OPENROUTER_API_KEY to use a hosted OpenAI-compatible model.
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

LABELS = {
    "Area": "area density code (A = rural to F = urban)", "VehGas": "fuel type", "Region": "French region code",
    "DrivAgeBand": "driver age band", "VehAgeBand": "vehicle age band (years)", "BonusMalusBand": "bonus-malus level (50 = best)",
}

SYSTEM = (
    "You write short notes for insurance pricing analysts. Use only numbers that appear in the table. "
    "Do not compute new numbers, do not add percentages that are not in the table, do not speculate about causes. "
    "Write 3 sentences, plain language, no lists, no headings."
)


def prompt_for(dim: str, rows: list[dict]) -> str:
    cols = PROMPT_COLS
    table = "\n".join(" | ".join(str(r[c]) for c in cols) for r in rows)
    return (
        f"Segment table by {LABELS[dim]} on a holdout of French motor policies. Columns: {' | '.join(cols)}.\n"
        "frequency = claims per policy-year, pure_premium = expected loss per policy-year in euros, "
        "lgb_relativity = segment premium divided by the portfolio average.\n\n"
        f"{table}\n\n"
        "Write the note: which segments carry the highest and lowest lgb_relativity, and for each of those two "
        "segments copy its model_fit value. Mention each segment in its own sentence."
    )


def call_llm(user: str, temperature: float = 0.2) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if PROVIDER == "ollama":
        body = {"model": MODEL, "messages": messages, "stream": False, "options": {"temperature": temperature, "seed": 7}}
        req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["message"]["content"].strip()
    if PROVIDER == "openrouter":
        body = {"model": MODEL, "messages": messages, "temperature": temperature}
        req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
    raise ValueError(f"unknown LLM_PROVIDER {PROVIDER}")


# Numbers not glued to a letter, digit or dot ("R82" and the "20" in "18-20" are labels, not claims).
NUM = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


PROMPT_COLS = ["segment", "exposure", "claims", "observed_frequency", "lgb_frequency",
               "observed_pure_premium", "lgb_pure_premium", "lgb_relativity", "model_fit"]


def model_fit(r: dict) -> str:
    """Direction of the model's frequency against observed, within 3% counts as close."""
    ratio = r["lgb_frequency"] / r["observed_frequency"] if r["observed_frequency"] else float("inf")
    return "close" if abs(ratio - 1) <= 0.03 else ("model above observed" if ratio > 1 else "model below observed")


def with_fit(rows: list[dict]) -> list[dict]:
    return [{**r, "model_fit": model_fit(r)} for r in rows]


def row_numbers(r: dict) -> set[float]:
    """Numbers an analyst could quote from one row of the prompt table."""
    vals = set()
    for k in PROMPT_COLS[1:-1]:
        v = float(r[k])
        vals.update({round(v, 4), round(v, 3), round(v, 2), round(v, 1), round(v, 0)})
        if k.endswith("frequency"):
            vals.update({round(v * 100, 2), round(v * 100, 1)})  # 0.2030 written as 20.3%
    vals.update(float(p) for p in re.findall(r"\d+", str(r["segment"])))
    return vals


def mentioned(sentence: str, rows: list[dict]) -> list[dict]:
    hits = []
    for r in rows:
        label = re.escape(str(r["segment"]))
        if re.search(rf"(?<![\w-]){label}(?![\w-])", sentence):
            hits.append(r)
    return hits


def check(note: str, rows: list[dict]) -> tuple[bool, list[str]]:
    """Return (ok, problems).

    1. Every number must come from the table, and from the row of a segment named in the
       same sentence when the sentence names one (so numbers cannot be swapped between rows).
    2. A sentence that says "highest" or "lowest" relativity must name the true segment.
    """
    problems = []
    all_vals = set().union(*(row_numbers(r) for r in rows))
    top = max(rows, key=lambda r: r["lgb_relativity"])
    bottom = min(rows, key=lambda r: r["lgb_relativity"])
    for sentence in re.split(r"(?<=[.!?])\s+", note):
        named = mentioned(sentence, rows)
        vals = set().union(*(row_numbers(r) for r in named)) if named else all_vals
        for tok in NUM.findall(sentence):
            x = float(tok.replace(",", ""))
            if x not in vals:
                problems.append(tok)
        # Direction claims are only checked when one segment is named, so they bind unambiguously.
        if len(named) == 1 and "model" in sentence.lower():
            fit = model_fit(named[0])
            low = sentence.lower()
            says_above = re.search(r"model[^.]*\b(higher|above|over)", low) is not None
            says_below = re.search(r"model[^.]*\b(lower|below|under)", low) is not None
            if (says_above and fit == "model below observed") or (says_below and fit == "model above observed"):
                problems.append(f"{named[0]['segment']}: {fit}")
        if "relativit" in sentence.lower() or "premium" in sentence.lower():
            low = sentence.lower()
            if "highest" in low and top not in named:
                problems.append(f"highest is {top['segment']}")
            if "lowest" in low and bottom not in named:
                problems.append(f"lowest is {bottom['segment']}")
    return (not problems), problems


def draft(dim: str, rows: list[dict], tries: int = 3) -> dict:
    rows = with_fit(rows)
    user = prompt_for(dim, rows)
    attempts = []
    for i in range(tries):
        text = call_llm(user, temperature=0.2 + 0.3 * i)
        ok, bad = check(text, rows)
        attempts.append({"text": text, "ok": ok, "problems": bad})
        if ok:
            break
    final = attempts[-1]
    return {"dimension": dim, "provider": PROVIDER, "model": MODEL, "verified": final["ok"],
            "note": final["text"] if final["ok"] else None, "attempts": attempts}


def main(art: str = "artifacts") -> list[dict]:
    segs = json.loads((Path(art) / "segments.json").read_text())
    notes = [draft(dim, rows) for dim, rows in segs.items()]
    (Path(art) / "notes.json").write_text(json.dumps(notes, indent=1))
    for n in notes:
        print(n["dimension"], "verified" if n["verified"] else "REJECTED", [a["problems"] for a in n["attempts"]])
    return notes


if __name__ == "__main__":
    main(*sys.argv[1:])
