"""One command that must pass before a change is called done.

There is no CI in this tree and no test suite; what there is instead are three
invariants that fail *silently* when broken, which is the dangerous kind:

  * the mirrors of evaluator logic in `src/catalog.py` — a `card_slots` drift
    turns the card-signature bonus into a no-op and quietly gives back +0.017,
    with no error anywhere (`tools/verify_mirror.py` says why);
  * the competition kit staying read-only — an edit there scores locally and
    fails on the organisers' copy;
  * the official composite, which is the only number that matters.

    python -m tools.check            # mirrors + kit + full harness run
    python -m tools.check --fast     # mirrors on a sample, kit only

The composite is asserted against EXPECTED_COMPOSITE, not printed for a human
to eyeball. A drop is a regression; a rise is a result nobody has written down
yet. Both fail, and both want reading.

Exits non-zero on the first failed invariant, so it is usable as a pre-commit
hook or a CI step the day either exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sim import KIT_ROOT  # noqa: E402

EXPECTED_COMPOSITE = 0.9626
TOLERANCE = 0.0005
OFFICIAL = ROOT / "results" / "official.json"


def run(label: str, argv: list[str]) -> bool:
    print(f"\n=== {label} ===", flush=True)
    return subprocess.run(argv, cwd=ROOT).returncode == 0


def kit_is_clean() -> bool:
    """The kit has no `.git` of its own, so `git -C` reports the PARENT repo and
    always looks clean. Scoping the path to the working tree is what actually
    asks the question."""
    print("\n=== competition kit unmodified ===", flush=True)
    try:
        out = subprocess.run(["git", "status", "--short", "--", KIT_ROOT.name],
                             cwd=KIT_ROOT.parent, capture_output=True,
                             text=True, timeout=30)
    except Exception as exc:
        print(f"  SKIP: cannot ask git ({exc})")
        return True
    if out.returncode != 0:
        print("  SKIP: not a git repository")
        return True
    dirty = out.stdout.strip()
    print(f"  {KIT_ROOT.name}: {'MODIFIED' if dirty else 'clean'}")
    if dirty:
        print(dirty)
    return not dirty


def score_is_expected() -> bool:
    """Read the composite `harness run` just wrote, and hold it to the number
    the tree is documented at."""
    try:
        got = float(json.loads(OFFICIAL.read_text(encoding="utf-8"))
                    ["recommended_technical_score"])
    except Exception as exc:
        print(f"\n  cannot read {OFFICIAL.name}: {exc}")
        return False
    delta = got - EXPECTED_COMPOSITE
    ok = abs(delta) <= TOLERANCE
    print(f"\n  composite {got:.4f}  expected {EXPECTED_COMPOSITE:.4f}  "
          f"delta {delta:+.4f}  {'OK' if ok else 'OUT OF TOLERANCE'}")
    if not ok:
        print("  (if this is a real improvement, update EXPECTED_COMPOSITE and "
              "the docs that quote it)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="sample the mirrors and skip the harness")
    args = parser.parse_args()

    python = sys.executable
    mirror = [python, "-m", "tools.verify_mirror"]
    if args.fast:
        mirror += ["-n", "2000"]

    ok = run("evaluator mirrors", mirror)
    ok &= kit_is_clean()
    if not args.fast:
        ok &= run("official score", [python, "-m", "tools.harness", "run"])
        ok &= score_is_expected()

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
