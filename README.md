# LLM Hallucination Detection System
##  Overview

This project detects hallucinations in LLM-generated answers using contextual verification.
Given a reference paragraph, question, and model answer, the system identifies inconsistencies and flags factual errors or contradictions.

## Hallucination Types Detected

* **1A – Contradiction of Source**
* **2A – Factual Inaccuracy**
The framework is modular and can be extended to support additional hallucination categories.

## Approach

* Context–answer comparison
* Entity and numeric consistency checking
* Structured JSON output for explainability

Example output:

```json
{
  "hallucination_types": ["1A", "2A"],
  "explanation": "...",
  "corrected_answer": "..."
}
```
## Sample Case

**Question:** When was Harvard University founded?
**Model Answer:** 1638
**Correct Answer (from context):** 1636

The system detects contradiction and factual error, then returns the corrected value.


## 🚀 Future Enhancements

* Add NLI-based contradiction detection
* Integrate semantic similarity scoring
* Use transformer models (BERT / DeBERTa)
* Add evaluation metrics (Precision, Recall, F1)


Add semantic similarity scoring

Use BERT / DeBERTa for contradiction detection
