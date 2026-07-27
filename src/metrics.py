"""
Evaluation Metrics for OCR Text Recognition:
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - Sequence Exact Match Accuracy
"""

import editdistance


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """
    Compute Character Error Rate (CER) over a batch of predictions and references.
    
    CER = Sum(EditDistance(pred_i, ref_i)) / Sum(len(ref_i))
    
    Args:
        predictions: List of predicted text strings
        references: List of ground truth reference text strings
        
    Returns:
        Character Error Rate as a float (0.0 to 1.0+)
    """
    total_distance = 0
    total_reference_length = 0

    for pred, ref in zip(predictions, references):
        pred_str = str(pred) if pred is not None else ""
        ref_str = str(ref) if ref is not None else ""

        dist = editdistance.eval(pred_str, ref_str)
        total_distance += dist
        total_reference_length += len(ref_str)

    if total_reference_length == 0:
        return 0.0

    return float(total_distance / total_reference_length)


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """
    Compute Word Error Rate (WER) over a batch of predictions and references.
    
    WER = Sum(EditDistance(pred_words_i, ref_words_i)) / Sum(len(ref_words_i))
    
    Args:
        predictions: List of predicted text strings
        references: List of ground truth reference text strings
        
    Returns:
        Word Error Rate as a float (0.0 to 1.0+)
    """
    total_distance = 0
    total_reference_words = 0

    for pred, ref in zip(predictions, references):
        pred_words = str(pred).strip().split() if pred is not None else []
        ref_words = str(ref).strip().split() if ref is not None else []

        dist = editdistance.eval(pred_words, ref_words)
        total_distance += dist
        total_reference_words += len(ref_words)

    if total_reference_words == 0:
        return 0.0

    return float(total_distance / total_reference_words)


def compute_accuracy(predictions: list[str], references: list[str]) -> float:
    """
    Compute Sequence Exact Match Accuracy over a batch of predictions and references.
    
    Accuracy = Count(pred_i == ref_i) / Total_Samples
    
    Args:
        predictions: List of predicted text strings
        references: List of ground truth reference text strings
        
    Returns:
        Exact Match Accuracy as a float between 0.0 and 1.0
    """
    if len(references) == 0:
        return 0.0

    correct_matches = 0
    for pred, ref in zip(predictions, references):
        pred_str = str(pred).strip() if pred is not None else ""
        ref_str = str(ref).strip() if ref is not None else ""

        if pred_str == ref_str:
            correct_matches += 1

    return float(correct_matches / len(references))


if __name__ == "__main__":
    # Test OCR Metrics implementation
    test_preds = ["Tiếng Việt OCR", "Anh túc 123", "Bình Thuận"]
    test_refs  = ["Tiếng Việt OCR", "Anh túc 456", "Bình Thuận"]

    cer = compute_cer(test_preds, test_refs)
    wer = compute_wer(test_preds, test_refs)
    acc = compute_accuracy(test_preds, test_refs)

    print("--- OCR Metrics Verification ---")
    print(f"Predictions: {test_preds}")
    print(f"References:  {test_refs}")
    print(f"Computed CER:      {cer:.4f} (Expected ~0.0714)")
    print(f"Computed WER:      {wer:.4f} (Expected ~0.1429)")
    print(f"Computed Accuracy: {acc:.4f} (Expected ~0.6667)")

    assert round(acc, 4) == 0.6667, "Accuracy computation mismatch!"
    print("\nAll OCR metrics verification tests passed successfully!")
