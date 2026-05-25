"""
Evaluation utilities and metrics.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


def _subset_style_lines(subset_name: Optional[str] = None, variant: str = "default") -> List[str]:
    if subset_name == "outdated":
        return [
            "Answer with a single word or short phrase only.",
            "Do not hedge or give multiple answers.",
        ]
    if subset_name == "longtail":
        return [
            "Answer with the most precise and fine-grained entity or value you can identify.",
            "Use a single word or short phrase.",
            "Do not hedge or give multiple answers.",
        ]
    if subset_name == "anchor_induced" and variant in {"ungrounded", "grounded"}:
        return [
            "Answer the question as written.",
            "Use a single word or short phrase.",
            "Do not hedge or give multiple answers.",
        ]
    if subset_name in {"head_tail_rarity", "head-tail-rarity"}:
        return [
            "Answer the question as written.",
            "Use a single word or short phrase.",
            "Do not hedge or give multiple answers.",
        ]
    if subset_name == "near_true":
        return [
            "Answer the question as written.",
            "Be concise and specific.",
            "Do not hedge or give multiple answers.",
        ]
    return [
        "Answer with a single word or short phrase only.",
        "Do not explain, hedge, or give multiple answers.",
    ]


def build_answer_prompt(
    question: str,
    subset_name: Optional[str] = None,
    variant: str = "default",
    prompt_condition: str = "baseline",
) -> str:
    """Build the answer prompt for a question."""
    style = _subset_style_lines(subset_name=subset_name, variant=variant)

    if prompt_condition == "zero_shot_EBP":
        style.extend(
            [
                "Let's think step by step.",
                "After reasoning through this, provide your final answer on a line that begins with FINAL ANSWER:",
            ]
        )
    elif prompt_condition == "boundary_aware_EBP":
        style.extend(
            [
                "Before answering, reason through the following:",
                "1. Do I have reliable knowledge about the specific entities or events in this question?",
                "2. Is this fact likely to be well-represented in my training data, or could it be obscure or rare?",
                "3. Could the premise of this question be false or misleading?",
                "After this reflection, provide your final answer on a line that begins with FINAL ANSWER:",
            ]
        )
    elif prompt_condition != "baseline":
        raise ValueError(f"Unknown prompt_condition: {prompt_condition}")

    if prompt_condition == "baseline":
        return "\n".join(style) + f"\n\nQuestion: {question}\nAnswer:"

    return "\n".join(style) + f"\n\nQuestion: {question}\nReasoning:\nFINAL ANSWER:"


def build_confidence_prompt(question: str, answer: str) -> str:
    """Build the confidence prompt for a question-answer pair."""
    return f"""You just answered the following question: "{question}"
Your answer was: "{answer}"

On a scale from 0 to 100, how confident are you that your answer is correct?
Reply with a single integer between 0 and 100. Nothing else."""


def build_answer_with_confidence_prompt(question: str, subset_name: Optional[str] = None, variant: str = "default") -> str:
    """Build a single-pass prompt that asks for both the answer and a confidence score."""
    if subset_name == "outdated":
        style = [
            "Answer with a single word or short phrase only.",
            "Do not hedge or give multiple answers.",
        ]
    elif subset_name == "longtail":
        style = [
            "Answer with the most precise and fine-grained entity or value you can identify.",
            "Use a single word or short phrase.",
            "Do not hedge or give multiple answers.",
        ]
    elif subset_name == "anchor_induced" and variant in {"grounded", "ungrounded"}:
        style = [
            "Answer the question as written.",
            "Use a single word or short phrase.",
            "Do not hedge or give multiple answers.",
        ]
    elif subset_name in {"head_tail_rarity", "head-tail-rarity"}:
        style = [
            "Answer the question as written.",
            "Use a single word or short phrase.",
            "Do not hedge or give multiple answers.",
        ]
    elif subset_name == "near_true":
        style = [
            "Answer the question as written.",
            "Be concise and specific.",
            "Do not hedge or give multiple answers.",
        ]
    else:
        style = [
            "Answer with a single word or short phrase only.",
            "Do not explain, hedge, or give multiple answers.",
        ]

    return (
        "\n".join(style)
        + f"\n\nQuestion: {question}\n"
        + "Return only valid JSON with exactly these keys: answer, confidence.\n"
        + 'answer should be your short answer as a string. confidence should be an integer from 0 to 100.\n'
        + 'Example: {"answer":"Paris","confidence":87}'
    )


def extract_final_answer(text: str) -> str:
    """Extract the final answer from a response that may include chain-of-thought."""
    if not text:
        return ""

    stripped = text.strip()
    if not stripped:
        return ""

    final_marker = re.search(r"(?im)^\s*FINAL ANSWER\s*:\s*(.+?)\s*$", stripped)
    if final_marker:
        return final_marker.group(1).strip()

    answer_marker = re.search(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$", stripped)
    if answer_marker:
        return answer_marker.group(1).strip()

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        tail = lines[-1]
        if tail.lower().startswith("confidence") and len(lines) > 1:
            return lines[-2]
        return tail

    return stripped


def is_refusal_text(text: str) -> bool:
    """Detect common refusal patterns in a response."""
    text_lower = text.lower().strip()
    refusal_patterns = [
        r"\bi don't know\b",
        r"\bi cannot\b",
        r"\bi can't\b",
        r"\bi'm unable\b",
        r"\bi don't have\b",
        r"\bi don't possess\b",
        r"\bunclear\b",
        r"\bdon't understand\b",
        r"\bi'm sorry\b",
        r"\bapologi[sz]e\b",
        r"\bcannot answer\b",
        r"\bnot enough information\b",
        r"\bi refuse\b",
    ]
    return any(re.search(pattern, text_lower) for pattern in refusal_patterns)


def is_deflection_text(text: str) -> bool:
    """Detect when the model rejects the premise and pivots away from the asked answer."""
    text_lower = text.lower().strip()
    deflection_patterns = [
        r"\byour question is wrong\b",
        r"\bthe question is wrong\b",
        r"\bthe premise is wrong\b",
        r"\bfalse premise\b",
        r"\bincorrect premise\b",
        r"\bthe prompt is wrong\b",
        r"\bthe statement is wrong\b",
        r"\bthis is not correct\b",
        r"\bnot accurate\b",
    ]
    return any(re.search(pattern, text_lower) for pattern in deflection_patterns)


def is_malformed_text(text: str, max_words: int = 15) -> bool:
    """Detect repeated-token or nonsensical outputs that should be treated as malformed."""
    if not text or not text.strip():
        return True

    stripped = text.strip()
    word_count = len(stripped.split())
    if word_count > max_words:
        return True

    if re.search(r"(\b\w+\b)(?:\s+\1){2,}", stripped, flags=re.IGNORECASE):
        return True

    if re.search(r"([!@#$%^&*()_+=\[\]{}|;:'\",.<>/?\\-])\1{3,}", stripped):
        return True

    if re.search(r"(.)\1{6,}", stripped):
        return True

    alpha_num_chars = sum(ch.isalnum() for ch in stripped)
    non_space_chars = sum(not ch.isspace() for ch in stripped)
    if non_space_chars and alpha_num_chars / non_space_chars < 0.35:
        return True

    return False


def classify_response(text: str, max_words: int = 15) -> Tuple[str, Dict[str, bool]]:
    """Classify a model response for downstream majority voting and tracking."""
    stripped = extract_final_answer(text).strip()
    flags = {
        "is_refusal": is_refusal_text(stripped),
        "is_deflection": is_deflection_text(stripped),
        "is_malformed": is_malformed_text(stripped, max_words=max_words),
    }

    if flags["is_malformed"]:
        return "malformed", flags
    if flags["is_refusal"]:
        return "refusal", flags
    return normalize_text(stripped), flags


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    
    # Alias expansion (before other normalization)
    aliases = {
        # Countries
        "usa": "united states",
        "u.s.a": "united states",
        "u.s.a.": "united states",
        "u s a": "united states",
        "us": "united states",
        "u k": "united kingdom",
        "gb": "united kingdom",
        "ussr": "soviet union",
        "soviet union": "soviet union",
        
        # Places
        "saint": "st",
        "st.": "st",
        
        # Common variants
        "los angeles": "los angeles",
        "new york": "new york",
        "san francisco": "san francisco",
    }
    
    for old, new in aliases.items():
        # Use word boundaries to avoid partial matches
        text = re.sub(r'\b' + re.escape(old) + r'\b', new, text)
    
    # Remove punctuation but keep hyphens for compound words
    text = re.sub(r"[^\w\s\-]", '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def is_valid_answer(text: str, max_words: int = 15) -> bool:
    """
    Check if answer is valid (not too long, not malformed).
    
    Args:
        text: The answer text to validate
        max_words: Maximum allowed words (default: 15)
    """
    if not text or len(text.strip()) < 2:
        return False
    
    # Too long — probably a hedged multi-part answer or explanation
    word_count = len(text.split())
    if word_count > max_words:
        logger.debug(f"Answer too long ({word_count} words, limit={max_words}): {text[:50]}")
        return False
    
    # Non-answer generic titles or refusals
    non_answers = {
        "mayor", "president", "prime minister", "minister",
        "unknown", "n/a", "none", "the president", "the mayor",
        "i don't know", "i cannot", "i can't", "cannot answer",
        "don't know", "unclear", "not available",
    }
    if text.lower().strip() in non_answers:
        return False
    
    return True


def parse_verbalized_confidence(raw_response: str) -> int:
    """Extract confidence score from model response."""
    match = re.search(r'\b(\d{1,3})\b', raw_response)
    if match:
        score = int(match.group(1))
        return min(max(score, 0), 100)  # clamp to [0, 100]
    return None


def determine_confusion_cell(is_confident: bool, is_correct: bool) -> str:
    """Determine which cell of the confusion matrix this belongs to."""
    if is_confident and is_correct:
        return "confident_correct"
    elif is_confident and not is_correct:
        return "confident_wrong"
    elif not is_confident and is_correct:
        return "uncertain_correct"
    else:
        return "uncertain_wrong"


class MetricsComputer:
    """Compute evaluation metrics (BCS, etc.)."""

    @staticmethod
    def compute_bcs(results: List[Dict], lambda_penalty: float = 2.0) -> Dict:
        """Compute Boundary Calibration Score."""
        counts = Counter(r["confusion_cell"] for r in results)
        N = len(results)

        if N == 0:
            return {"bcs": 0, "n": 0}

        cc = counts.get("confident_correct", 0)
        cw = counts.get("confident_wrong", 0)
        uc = counts.get("uncertain_correct", 0)
        uw = counts.get("uncertain_wrong", 0)

        bcs = (cc + uw) / N - lambda_penalty * (cw / N)

        return {
            "bcs": round(bcs, 4),
            "n": N,
            "confident_correct": cc,
            "confident_wrong": cw,
            "uncertain_correct": uc,
            "uncertain_wrong": uw,
            "cc_pct": round(cc / N, 3),
            "cw_pct": round(cw / N, 3),
            "uc_pct": round(uc / N, 3),
            "uw_pct": round(uw / N, 3),
        }

    @staticmethod
    def compute_baselines(lambda_penalty: float = 2.0) -> Dict:
        """
        Compute baseline BCS scores for context.
        
        Returns:
            Dictionary with baseline BCS values for different scenarios
        """
        # Random baseline: model is always 50% confident and 50% correct
        # CC: 25%, CW: 25%, UC: 25%, UW: 25%
        random_bcs = (0.25 + 0.25) - lambda_penalty * 0.25
        
        # Always-uncertain baseline: model never confident
        # UC: 50%, UW: 50%
        uncertain_bcs = 0.50
        
        # Always-wrong baseline: model is always confident but always wrong
        # CW: 100%
        always_wrong_bcs = 0 - lambda_penalty * 1.0
        
        return {
            "random_guess": round(random_bcs, 4),
            "always_uncertain": round(uncertain_bcs, 4),
            "always_wrong": round(always_wrong_bcs, 4),
            "perfect": 1.0,  # 100% CC
        }

    @staticmethod
    def print_metrics(metrics: Dict, model_name: str = "", subset_name: str = ""):
        """Pretty-print metrics."""
        label = f"{model_name} | {subset_name}" if model_name else subset_name
        logger.info(f"\n=== BCS Metrics: {label} ===")
        logger.info(f"N = {metrics['n']}")
        logger.info(f"Confident-Correct:  {metrics['confident_correct']:3d} ({metrics['cc_pct']:5.1%})")
        logger.info(f"Confident-Wrong:    {metrics['confident_wrong']:3d} ({metrics['cw_pct']:5.1%})")
        logger.info(f"Uncertain-Correct:  {metrics['uncertain_correct']:3d} ({metrics['uc_pct']:5.1%})")
        logger.info(f"Uncertain-Wrong:    {metrics['uncertain_wrong']:3d} ({metrics['uw_pct']:5.1%})")
        logger.info(f"BCS = {metrics['bcs']:.4f}")
        
        # Show baselines for context
        baselines = MetricsComputer.compute_baselines()
        logger.info(f"\nContext (baselines):")
        logger.info(f"  Perfect model:     BCS = {baselines['perfect']:.4f}")
        logger.info(f"  Random 50% guess:  BCS = {baselines['random_guess']:.4f}")
        logger.info(f"  Always uncertain:  BCS = {baselines['always_uncertain']:.4f}")
        logger.info(f"  Always wrong:      BCS = {baselines['always_wrong']:.4f}")
