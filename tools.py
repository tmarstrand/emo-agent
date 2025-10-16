"""
Tools for ADK agents to analyze emotional content using VAD (Valence, Arousal, Dominance) model.
"""
from main import VADPredictor

# Initialize the predictor once (singleton pattern for efficiency)
_predictor = None

def get_predictor() -> VADPredictor:
    """Lazy load the VAD predictor."""
    global _predictor
    if _predictor is None:
        _predictor = VADPredictor()
    return _predictor


def analyze_user_emotion(text: str) -> dict[str, float]:
    """
    Analyze the emotional content of text using VAD (Valence, Arousal, Dominance) model.

    This tool helps you understand the user's emotional state by mapping their text
    to a three-dimensional emotional space:

    - Valence: Measures positivity/negativity (range typically -3 to +3)
      * Positive values = pleasant, happy, satisfied
      * Negative values = unpleasant, unhappy, dissatisfied

    - Arousal: Measures activation/energy level (range typically 1 to 5)
      * High values = excited, stimulated, alert
      * Low values = calm, relaxed, sluggish

    - Dominance: Measures control/power (range typically 1 to 5)
      * High values = in control, influential, dominant
      * Low values = controlled, influenced, submissive

    Args:
        text: The user's message to analyze

    Returns:
        Dictionary with keys 'valence', 'arousal', 'dominance' and their corresponding scores.

    Example:
        >>> analyze_user_emotion("I am so happy and excited!")
        {'valence': 2.8, 'arousal': 4.2, 'dominance': 3.5}
    """
    predictor = get_predictor()
    return predictor.predict(text)
