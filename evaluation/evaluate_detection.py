
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.label_reader import read_document
from tools.change_detector import extract_changed_sections


DATASET_PATH = PROJECT_ROOT / "evaluation" / "ground_truth.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_pair(pair):
    old_path = PROJECT_ROOT / pair["old_label"]
    new_path = PROJECT_ROOT / pair["new_label"]

    old_text = read_document(str(old_path))
    new_text = read_document(str(new_path))

    detected = extract_changed_sections(old_text, new_text)

    ground_truth = pair["ground_truth"]

    expected_sections = {
        item["section"].casefold()
        for item in ground_truth["changed_sections"]
    }

    detected_sections = {
        item["section"].casefold()
        for item in detected
    }

    section_universe = {
        section.casefold()
        for section in ground_truth["unchanged_sections"]
    } | expected_sections | detected_sections

    tp = len(expected_sections & detected_sections)
    fp = len(detected_sections - expected_sections)
    fn = len(expected_sections - detected_sections)
    tn = len(
        section_universe
        - expected_sections
        - detected_sections
    )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    accuracy = (
        (tp + tn) / len(section_universe)
        if section_universe else 0.0
    )

    exact_match = expected_sections == detected_sections

    print(f"\nEvaluation: {pair['pair_id']}")
    print("=" * 55)

    print("Expected changes:", sorted(expected_sections))
    print("Detected changes:", sorted(detected_sections))

    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives:  {tn}")

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    print("Exact section match:", exact_match)

    return {
        "pair_id": pair["pair_id"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "exact_match": exact_match
    }


def main():
    dataset = load_dataset()

    print(dataset["dataset_name"])
    print(dataset["disclaimer"])

    results = []

    for pair in dataset["evaluation_pairs"]:
        results.append(evaluate_pair(pair))

    total_tp = sum(result["tp"] for result in results)
    total_fp = sum(result["fp"] for result in results)
    total_fn = sum(result["fn"] for result in results)
    total_tn = sum(result["tn"] for result in results)

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    total = total_tp + total_fp + total_fn + total_tn
    accuracy = (
        (total_tp + total_tn) / total
        if total else 0.0
    )

    print("\n\nOVERALL EVALUATION")
    print("=" * 55)
    print("Evaluation pairs:", len(results))
    print("TP:", total_tp)
    print("FP:", total_fp)
    print("FN:", total_fn)
    print("TN:", total_tn)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    exact_matches = sum(
        result["exact_match"] for result in results
    )

    print(
        "Pair-level exact match:",
        f"{exact_matches}/{len(results)}"
    )


if __name__ == "__main__":
    main()
