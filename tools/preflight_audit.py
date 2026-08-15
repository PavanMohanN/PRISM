"""PRISM Phase 6 -- preflight submission audit (guide items 25, 32, + all wording).

Mechanically enforces the revision before submission. Scans the assembled .tex and:

  ERRORS (must fix):
    * stale phrases that the revision removed
      ("machine precision", "exact forward-inverse consistency", "solver tolerance",
       "ICLR 2026", ...)
    * \\ref/\\eqref to a label that does not exist
    * \\input / \\includegraphics pointing at a file that is missing
    * the SLCP-modality contradiction (calling SLCP both multimodal and single-mode)

  WARNINGS (manual check): every "best", "only", "exact", "guaranteed" claim, so a
  human confirms each is supported by a table before submission.

Run:  python tools/preflight_audit.py prism_iclr2027.tex --root .
"""
from __future__ import annotations

import os
import re
import sys
import argparse

STALE_DEFAULT = [
    "machine precision",
    "exact forward-inverse consistency",
    "exact forward inverse consistency",
    "exact consistency",
    "solver tolerance",
    "best calibrated",
    "best-calibrated",
    "iclr 2026",
]
REVIEW_TERMS = ["best", "only", "exact", "guaranteed", "state-of-the-art", "sota"]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def audit_text(tex, root=".", stale=None, review=None):
    stale = stale if stale is not None else STALE_DEFAULT
    review = review if review is not None else REVIEW_TERMS
    errors, warnings = [], []
    low = tex.lower()

    # 1. stale phrases
    for term in stale:
        if term.lower() in low:
            n = low.count(term.lower())
            errors.append(f"stale phrase present x{n}: '{term}'")

    # 2. unresolved references (expand \input files one level to gather labels)
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    for inc in re.findall(r"\\input\{([^}]*)\}", tex):
        for cand in (inc, inc + ".tex"):
            p = os.path.join(root, cand)
            if os.path.exists(p):
                labels |= set(re.findall(r"\\label\{([^}]*)\}", _read(p)))
                break
    refs = re.findall(r"\\(?:ref|eqref|autoref|cref)\{([^}]*)\}", tex)
    for r in refs:
        for part in r.split(","):
            part = part.strip()
            if part and part not in labels:
                errors.append(f"unresolved reference: \\ref{{{part}}}")

    # 3. missing \input / \includegraphics targets
    for inc in re.findall(r"\\input\{([^}]*)\}", tex):
        cands = [inc, inc + ".tex"]
        if not any(os.path.exists(os.path.join(root, c)) for c in cands):
            errors.append(f"missing \\input file: {inc}")
    for img in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        cands = [img] + [img + e for e in (".pdf", ".png", ".jpg")]
        if not any(os.path.exists(os.path.join(root, c)) for c in cands):
            errors.append(f"missing figure file: {img}")

    # 4. SLCP modality contradiction (stop at clause boundaries; multimodal wording only)
    slcp_multi = bool(re.search(r"slcp[^.;]{0,60}(multimodal|multi-modal)", low)) or \
        bool(re.search(r"(multimodal|multi-modal)[^.;]{0,60}slcp", low))
    slcp_single = bool(re.search(r"slcp[^.;]{0,60}(single[- ]dominant|single mode|unimodal)", low)) or \
        bool(re.search(r"(single[- ]dominant|single mode|unimodal)[^.;]{0,60}slcp", low))
    if slcp_multi and slcp_single:
        errors.append("SLCP modality contradiction: described as BOTH multimodal and single-mode")

    # 5. review-term warnings (with line numbers)
    for i, line in enumerate(tex.splitlines(), 1):
        ll = line.lower()
        for t in review:
            if re.search(rf"\b{re.escape(t)}\b", ll):
                warnings.append(f"L{i}: review '{t}' -> confirm supported by a table: {line.strip()[:80]}")

    return {"errors": errors, "warnings": warnings, "ok": len(errors) == 0}


def audit_file(path, root="."):
    return audit_text(_read(path), root=root)


def _print(report):
    print(f"\n=== PRISM preflight audit ===")
    print(f"errors:   {len(report['errors'])}")
    for e in report["errors"]:
        print(f"  [ERROR] {e}")
    print(f"warnings: {len(report['warnings'])} (manual check)")
    for w in report["warnings"][:40]:
        print(f"  [warn] {w}")
    if len(report["warnings"]) > 40:
        print(f"  ... and {len(report['warnings']) - 40} more")
    print("\nRESULT:", "PASS" if report["ok"] else "FAIL (fix errors above)")
    return report["ok"]


def _selftest():
    good = r"""
    \label{tab:x}\label{eq:y}
    We report structurally reversible latent transport. See Table~\ref{tab:x} and Eq.~\eqref{eq:y}.
    SLCP has a single-dominant mode; Two Moons is disconnected.
    """
    bad = r"""
    We prove exact forward-inverse consistency and machine precision reversibility.
    SLCP is multimodal. Later: SLCP has a single mode.
    See Table~\ref{tab:missing}. Submitted to ICLR 2026.
    """
    g = audit_text(good, root=".", review=[])   # suppress review warns for the clean case
    b = audit_text(bad, root=".")
    assert g["ok"], g["errors"]
    assert not b["ok"]
    joined = " ".join(b["errors"]).lower()
    assert "machine precision" in joined
    assert "exact forward-inverse consistency" in joined
    assert "iclr 2026" in joined
    assert "unresolved reference" in joined
    assert "contradiction" in joined
    print("preflight self-test: PASS")
    print(f"  clean doc -> {len(g['errors'])} errors")
    print(f"  planted doc -> {len(b['errors'])} errors caught")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", nargs="?", help="manuscript .tex to audit")
    ap.add_argument("--root", default=".")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest or not args.tex:
        _selftest()
    else:
        ok = _print(audit_file(args.tex, root=args.root))
        sys.exit(0 if ok else 1)
