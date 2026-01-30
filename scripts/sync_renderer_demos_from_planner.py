#!/usr/bin/env python3
from __future__ import annotations

"""
Keep renderer demo inputs aligned with the planner demo outputs.

Why:
- The renderer demos take `question_plan_json` as input.
- That plan should be sourced from the planner demos (single source of truth),
  otherwise fields drift (e.g. `option_hints` vs old `answer_hints`) and examples
  stop composing correctly.

What it does:
- For each renderer demo example, finds a planner demo whose `plan[]` contains
  all keys referenced by the renderer demo.
- Rebuilds the renderer demo's `inputs.question_plan_json.plan[]` by pulling the
  corresponding plan items from the planner demo output (preserving the renderer
  example's key order).
- Normalizes legacy aliases per plan item:
  - `answer_hints` -> `option_hints` (if `option_hints` missing)

Run:
  python3 scripts/sync_renderer_demos_from_planner.py --write
  python3 scripts/sync_renderer_demos_from_planner.py --check
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _norm_key(k: Any) -> str:
    return str(k or "").strip().lower()


def _extract_plan(obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    plan = obj.get("plan")
    if isinstance(plan, list):
        return [x for x in plan if isinstance(x, dict)]
    return []


def _normalize_plan_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    if "option_hints" not in out and "answer_hints" in out:
        out["option_hints"] = out.get("answer_hints")
    out.pop("answer_hints", None)
    return out


def _planner_index(planner_examples: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
    """
    Returns a list of (planner_example, key->plan_item) for matching.
    """
    out: List[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]] = []
    for ex in planner_examples:
        outputs = ex.get("outputs") if isinstance(ex.get("outputs"), dict) else {}
        plan_obj = outputs.get("question_plan_json")
        plan = _extract_plan(plan_obj)
        by_key: Dict[str, Dict[str, Any]] = {}
        for it in plan:
            k = _norm_key(it.get("key"))
            if not k:
                continue
            by_key[k] = _normalize_plan_item(it)
        if by_key:
            out.append((ex, by_key))
    return out


def _find_best_planner_match(
    *,
    planner_idx: List[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]],
    needed_keys: List[str],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
    need = set([_norm_key(k) for k in needed_keys if _norm_key(k)])
    if not need:
        return None

    best = None
    best_extra = None
    for ex, by_key in planner_idx:
        keys = set(by_key.keys())
        if not need.issubset(keys):
            continue
        extra = len(keys - need)
        if best is None or (best_extra is not None and extra < best_extra):
            best = (ex, by_key)
            best_extra = extra
    return best


def _sync_renderer_example(
    *,
    renderer_ex: Dict[str, Any],
    planner_idx: List[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]],
) -> Tuple[bool, List[str]]:
    """
    Returns (changed, issues).
    """
    issues: List[str] = []
    inputs = renderer_ex.get("inputs") if isinstance(renderer_ex.get("inputs"), dict) else {}
    plan_obj = inputs.get("question_plan_json")
    plan = _extract_plan(plan_obj)
    needed_keys = [_norm_key(it.get("key")) for it in plan if isinstance(it, dict)]
    needed_keys = [k for k in needed_keys if k]
    if not needed_keys:
        issues.append("renderer example missing inputs.question_plan_json.plan[] keys")
        return (False, issues)

    match = _find_best_planner_match(planner_idx=planner_idx, needed_keys=needed_keys)
    if not match:
        issues.append(f"no planner demo contains keys={needed_keys}")
        return (False, issues)

    _, by_key = match
    new_plan: List[Dict[str, Any]] = []
    for it in plan:
        if not isinstance(it, dict):
            continue
        k = _norm_key(it.get("key"))
        if not k:
            continue
        base = copy.deepcopy(by_key.get(k) or {})
        if not base:
            issues.append(f"planner match missing key={k} (unexpected)")
            base = {}
        # Preserve any renderer-specific fields if present (rare).
        merged = dict(base)
        for rk, rv in it.items():
            if rk in ("answer_hints",):
                continue
            if rk not in merged and rv is not None:
                merged[rk] = rv
        new_plan.append(_normalize_plan_item(merged))

    new_inputs = dict(inputs)
    new_inputs["question_plan_json"] = {"plan": new_plan}

    changed = json.dumps(inputs.get("question_plan_json"), sort_keys=True) != json.dumps(
        new_inputs.get("question_plan_json"), sort_keys=True
    )
    if changed:
        renderer_ex["inputs"] = new_inputs
    return (changed, issues)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Fail if any renderer demo would change.")
    ap.add_argument("--write", action="store_true", help="Rewrite renderer demo file in-place.")
    ap.add_argument(
        "--planner-path",
        default=str(_repo_root() / "src" / "programs" / "question_planner" / "data" / "examples" / "demo_examples.json"),
    )
    ap.add_argument(
        "--renderer-path",
        default=str(_repo_root() / "src" / "programs" / "renderer" / "examples" / "demo_examples.json"),
    )
    args = ap.parse_args(argv)

    if not args.check and not args.write:
        raise SystemExit("Pass --check or --write.")

    planner_path = Path(args.planner_path)
    renderer_path = Path(args.renderer_path)

    planner_examples = _load_json(planner_path)
    renderer_examples = _load_json(renderer_path)
    if not isinstance(planner_examples, list) or not isinstance(renderer_examples, list):
        raise SystemExit("Expected both demo files to be JSON arrays.")

    planner_idx = _planner_index([x for x in planner_examples if isinstance(x, dict)])
    changed_any = False
    issues_any: List[str] = []

    for i, ex in enumerate([x for x in renderer_examples if isinstance(x, dict)]):
        changed, issues = _sync_renderer_example(renderer_ex=ex, planner_idx=planner_idx)
        if issues:
            issues_any.extend([f"renderer[{i}]: {msg}" for msg in issues])
        changed_any = changed_any or changed

    if args.check:
        if issues_any:
            for msg in issues_any:
                print(msg, file=sys.stderr)
            return 2
        if changed_any:
            print("renderer demos are out of sync with planner demos", file=sys.stderr)
            return 1
        print("ok: renderer demos already in sync")
        return 0

    if issues_any:
        for msg in issues_any:
            print(msg, file=sys.stderr)
        return 2
    if args.write and changed_any:
        _write_json(renderer_path, renderer_examples)
        print(f"updated: {renderer_path}")
    else:
        print("no changes needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
