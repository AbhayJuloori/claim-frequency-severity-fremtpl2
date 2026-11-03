"""Copy the holdout snapshot the UI reads when no API is configured."""
import json
import shutil
import sys
from pathlib import Path


def main(art: str = "artifacts", dest: str = "web/public/data") -> None:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(art) / "segments.json", out / "segments.json")
    m = json.loads((Path(art) / "metrics.json").read_text())
    (out / "metrics.json").write_text(json.dumps({k: m[k] for k in ["audit", "n_train", "n_test", "frequency", "severity", "pure_premium", "deciles"]}))
    notes = json.loads((Path(art) / "notes.json").read_text())
    (out / "notes.json").write_text(json.dumps([{"dimension": n["dimension"], "verified": n["verified"], "note": n["note"],
                                                 "model": n["model"], "problems": n["attempts"][-1]["problems"]} for n in notes]))


if __name__ == "__main__":
    main(*sys.argv[1:])
