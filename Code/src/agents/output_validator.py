def validate_llm_output(text: str) -> bool:
    """
    Basic validation to detect empty or hallucinated responses
    """
    if not text:
        return False

    banned_phrases = ["guaranteed", "certain profit", "risk-free"]
    return not any(p in text.lower() for p in banned_phrases)
