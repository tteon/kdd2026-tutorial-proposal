"""
Evaluate experiment artifacts on two axes:

  Axis 1 — Gold-based (circular, for debugging):  P/R/F1, linking accuracy
  Axis 2 — FIBO conformance (no gold needed):     ontology guide compliance

Usage:
    uv run python -m experiments.eval_smoke [--experiment rdf_vs_lpg]
    uv run python -m experiments.eval_smoke --experiment rdf_vs_lpg --opik

Reads:
  - artifacts/{experiment}/manifest.json
  - manual_gold/seed_gold.json

Outputs:
  - artifacts/{experiment}/evaluation.json
  - (with --opik) Opik Cloud experiment with scores visible in dashboard
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from framework.evaluation import (
    ExtractionScores,
    FIBOConformanceScores,
    LinkingScores,
    aggregate_conformance,
    aggregate_metrics,
    compute_self_consistency_metrics,
    evaluate_extraction,
    evaluate_fibo_conformance,
    evaluate_linking,
    run_opik_evaluate,
)
from manual_gold.gold_schema import load_gold_examples


def load_artifact(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-axis evaluation of experiment artifacts.")
    parser.add_argument("--experiment", default="rdf_vs_lpg",
                        help="Experiment name under artifacts/ (default: rdf_vs_lpg)")
    parser.add_argument("--opik", action="store_true",
                        help="Run Opik evaluate() and push scores to Opik Cloud dashboard")
    args = parser.parse_args()

    experiment_dir = Path("artifacts") / args.experiment
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"Experiment: {args.experiment}")
    print(f"Examples in manifest: {len(manifest['examples'])}")

    gold_examples = load_gold_examples("manual_gold/seed_gold.json")
    gold_by_id = {g.example_id: g for g in gold_examples}
    print(f"Gold: {len(gold_examples)} examples, {sum(len(g.entities) for g in gold_examples)} entities\n")

    # Collectors
    results: list[dict[str, Any]] = []
    rdf_ext: list[ExtractionScores] = []
    rdf_link: list[LinkingScores] = []
    lpg_ext: list[ExtractionScores] = []
    lpg_link: list[LinkingScores] = []
    rdf_conf: list[FIBOConformanceScores] = []
    lpg_conf: list[FIBOConformanceScores] = []

    # ========================================================================
    # Axis 1: Gold-based evaluation (per-example)
    # ========================================================================
    print("=" * 90)
    print("AXIS 1: Gold-based Evaluation (circular — for debugging only)")
    print("=" * 90)

    for entry in manifest["examples"]:
        example_id = entry["example_id"]
        rep = entry["representation"]
        category = entry["category"]
        artifact = load_artifact(entry["artifact_path"])

        gold = gold_by_id.get(example_id)
        if not gold:
            print(f"  SKIP {rep:3s} | {example_id} — no gold")
            continue

        predicted_entities = artifact.get("extracted_entities", [])
        ext_scores = evaluate_extraction(predicted_entities, gold.entity_texts())

        predicted_linked = artifact.get("linked_entities", [])
        link_scores = evaluate_linking(predicted_linked, gold.entity_concept_pairs())

        self_consist = compute_self_consistency_metrics(artifact)

        if rep == "rdf":
            rdf_ext.append(ext_scores)
            rdf_link.append(link_scores)
        else:
            lpg_ext.append(ext_scores)
            lpg_link.append(link_scores)

        result = {
            "example_id": example_id,
            "category": category,
            "representation": rep,
            "extraction": asdict(ext_scores),
            "linking": asdict(link_scores),
            "self_consistency": self_consist,
        }
        results.append(result)

        print(f"  {rep:3s} | {category:20s} | {example_id} | "
              f"P={ext_scores.precision:.2f} R={ext_scores.recall:.2f} F1={ext_scores.f1:.2f} | "
              f"Link={link_scores.accuracy:.2f} ({link_scores.correct}/{link_scores.total})")

    rdf_agg = aggregate_metrics(rdf_ext, rdf_link)
    lpg_agg = aggregate_metrics(lpg_ext, lpg_link)

    print(f"\n  RDF macro:  P={rdf_agg['extraction']['macro_precision']:.4f}  "
          f"R={rdf_agg['extraction']['macro_recall']:.4f}  "
          f"F1={rdf_agg['extraction']['macro_f1']:.4f}  "
          f"LinkAcc={rdf_agg['linking']['macro_accuracy']:.4f}")
    print(f"  LPG macro:  P={lpg_agg['extraction']['macro_precision']:.4f}  "
          f"R={lpg_agg['extraction']['macro_recall']:.4f}  "
          f"F1={lpg_agg['extraction']['macro_f1']:.4f}  "
          f"LinkAcc={lpg_agg['linking']['macro_accuracy']:.4f}")

    # ========================================================================
    # Axis 2: FIBO Conformance (no gold needed)
    # ========================================================================
    print(f"\n{'=' * 90}")
    print("AXIS 2: FIBO Ontology Guide Conformance (authority-grounded, no gold needed)")
    print("=" * 90)
    print(f"\n  Criteria derived from context-ONTOLOGY_GUIDE.md:")
    print(f"    - Entity types:  UpperCamelCase, singular, no abbreviations")
    print(f"    - Concept IDs:   FIBO namespace prefix, known domain, class CamelCase")
    print(f"    - Labels:        unique mapping (label → concept_id)")
    print(f"    - RDF triples:   predicate lowerCamelCase + FIBO prefix, typed S/O")
    print()

    for entry in manifest["examples"]:
        example_id = entry["example_id"]
        rep = entry["representation"]
        category = entry["category"]
        artifact = load_artifact(entry["artifact_path"])

        conf = evaluate_fibo_conformance(artifact)

        if rep == "rdf":
            rdf_conf.append(conf)
        else:
            lpg_conf.append(conf)

        # Attach to results
        for r in results:
            if r["example_id"] == example_id and r["representation"] == rep:
                r["fibo_conformance"] = asdict(conf)
                # Don't dump all violations into JSON — just count
                r["fibo_conformance"]["violation_count"] = len(conf.violations)
                r["fibo_conformance"]["violations"] = conf.violations[:5]  # top 5 only
                break

        violation_summary = f"{len(conf.violations)} violations" if conf.violations else "clean"
        print(f"  {rep:3s} | {category:20s} | {example_id} | "
              f"overall={conf.overall_conformance:.2f} | "
              f"etype_cc={conf.entity_type_camelcase_rate:.2f} "
              f"cid_pfx={conf.concept_id_has_prefix_rate:.2f} "
              f"cid_dom={conf.concept_id_known_domain_rate:.2f} "
              f"cid_cc={conf.concept_id_class_camelcase_rate:.2f}", end="")
        if conf.total_triples > 0:
            print(f" | pred_cc={conf.triple_predicate_camelcase_rate:.2f} "
                  f"pred_pfx={conf.triple_predicate_has_prefix_rate:.2f} "
                  f"subj_t={conf.triple_subject_typed_rate:.2f} "
                  f"obj_t={conf.triple_object_typed_rate:.2f}", end="")
        print(f" | {violation_summary}")

    rdf_conf_agg = aggregate_conformance(rdf_conf)
    lpg_conf_agg = aggregate_conformance(lpg_conf)

    def _fmt_conf(val: float | None) -> str:
        return f"{val:.4f}" if val is not None else "  N/A "

    print(f"\n  RDF conformance macro:  overall={rdf_conf_agg.get('macro_overall_conformance', 0):.4f}  "
          f"semantic={_fmt_conf(rdf_conf_agg.get('macro_semantic_conformance'))}  "
          f"structural={_fmt_conf(rdf_conf_agg.get('macro_structural_conformance'))}  "
          f"violations={rdf_conf_agg.get('avg_violations_per_artifact', 0):.1f}/artifact")
    print(f"  LPG conformance macro:  overall={lpg_conf_agg.get('macro_overall_conformance', 0):.4f}  "
          f"semantic={_fmt_conf(lpg_conf_agg.get('macro_semantic_conformance'))}  "
          f"structural={_fmt_conf(lpg_conf_agg.get('macro_structural_conformance'))}  "
          f"violations={lpg_conf_agg.get('avg_violations_per_artifact', 0):.1f}/artifact")

    # ========================================================================
    # Axis 2 detail: breakdown by conformance dimension
    # ========================================================================
    print(f"\n  {'Dimension':<40s} {'RDF':>8s} {'LPG':>8s}")
    print(f"  {'-'*56}")
    dimension_labels = {
        "macro_entity_type_camelcase_rate": "Entity type UpperCamelCase",
        "macro_entity_type_singular_rate": "Entity type singular",
        "macro_entity_type_no_abbreviation_rate": "Entity type no abbreviation",
        "macro_concept_id_has_prefix_rate": "Concept ID has FIBO prefix",
        "macro_concept_id_known_domain_rate": "Concept ID known domain",
        "macro_fibo_namespace_rate": "FIBO namespace (vs ama: ext)",
        "macro_concept_id_class_camelcase_rate": "Concept ID class CamelCase",
        "macro_concept_label_unique_rate": "Label uniqueness",
        "macro_semantic_conformance": "── Semantic (above dims) ──",
        "macro_triple_predicate_camelcase_rate": "Triple pred lowerCamelCase",
        "macro_triple_predicate_has_prefix_rate": "Triple pred FIBO prefix",
        "macro_triple_subject_typed_rate": "Triple subject typed",
        "macro_triple_object_typed_rate": "Triple object typed",
        "macro_structural_conformance": "── Structural (above dims) ──",
    }
    for key, label in dimension_labels.items():
        rdf_val = rdf_conf_agg.get(key)
        lpg_val = lpg_conf_agg.get(key)
        rdf_str = f"{rdf_val:8.4f}" if isinstance(rdf_val, (int, float)) else "     N/A"
        lpg_str = f"{lpg_val:8.4f}" if isinstance(lpg_val, (int, float)) else "     N/A"
        marker = ""
        if isinstance(rdf_val, (int, float)) and isinstance(lpg_val, (int, float)):
            if rdf_val > lpg_val + 0.01:
                marker = " ←RDF"
            elif lpg_val > rdf_val + 0.01:
                marker = " →LPG"
        print(f"  {label:<40s} {rdf_str} {lpg_str}{marker}")

    # ========================================================================
    # Category breakdown (F1 + conformance side by side)
    # ========================================================================
    print(f"\n{'=' * 90}")
    print("Category Breakdown (gold F1 + FIBO conformance)")
    print(f"{'Category':25s} {'RDF F1':>8s} {'LPG F1':>8s}  {'RDF Conf':>9s} {'LPG Conf':>9s}")
    print("-" * 70)

    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        rdf_r = [r for r in results if r["category"] == cat and r["representation"] == "rdf"]
        lpg_r = [r for r in results if r["category"] == cat and r["representation"] == "lpg"]
        rdf_f1 = rdf_r[0]["extraction"]["f1"] if rdf_r else 0
        lpg_f1 = lpg_r[0]["extraction"]["f1"] if lpg_r else 0
        rdf_c = rdf_r[0].get("fibo_conformance", {}).get("overall_conformance", 0) if rdf_r else 0
        lpg_c = lpg_r[0].get("fibo_conformance", {}).get("overall_conformance", 0) if lpg_r else 0
        print(f"{cat:25s} {rdf_f1:8.4f} {lpg_f1:8.4f}   {rdf_c:8.4f} {lpg_c:8.4f}")

    # ========================================================================
    # Top violations
    # ========================================================================
    print(f"\n{'=' * 90}")
    print("Top FIBO Conformance Violations (sample)")
    print("=" * 90)

    all_violations: dict[str, int] = {}
    for conf_list in [rdf_conf, lpg_conf]:
        for conf in conf_list:
            for v in conf.violations:
                # Generalize violation messages for counting
                key = v.split("'")[0].strip() if "'" in v else v
                all_violations[key] = all_violations.get(key, 0) + 1

    for violation, count in sorted(all_violations.items(), key=lambda x: -x[1])[:10]:
        print(f"  [{count:3d}x] {violation}")

    # ========================================================================
    # Save
    # ========================================================================
    eval_output = {
        "gold_example_count": len(gold_examples),
        "gold_entity_count": sum(len(g.entities) for g in gold_examples),
        "axis1_gold_based": {
            "rdf_aggregate": rdf_agg,
            "lpg_aggregate": lpg_agg,
            "note": "Circular evaluation — gold derived from pipeline output. Use for debugging only.",
        },
        "axis2_fibo_conformance": {
            "rdf_aggregate": rdf_conf_agg,
            "lpg_aggregate": lpg_conf_agg,
            "note": "Authority-grounded evaluation from FIBO Ontology Guide. No gold needed.",
        },
        "per_example": results,
    }
    eval_path = experiment_dir / "evaluation.json"
    eval_path.write_text(json.dumps(eval_output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {eval_path}")

    # ========================================================================
    # Opik evaluate() — push scores to Opik Cloud dashboard
    # ========================================================================
    if args.opik:
        print(f"\n{'=' * 90}")
        print("Opik evaluate() — pushing scores to dashboard")
        print("=" * 90)
        try:
            opik_result = run_opik_evaluate(
                experiment_name=args.experiment,
                manifest=manifest,
                gold_by_id=gold_by_id if gold_by_id else None,
                project_name="ama_evaluation",
            )
            print(f"  Experiment: {opik_result.experiment_name}")
            print(f"  Experiment URL: {opik_result.experiment_url}")
            print(f"  Test results: {len(opik_result.test_results)}")
            for es in opik_result.experiment_scores:
                print(f"  Experiment score: {es.name} = {es.value:.4f}")
        except Exception as exc:
            print(f"  Opik evaluate() failed: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
