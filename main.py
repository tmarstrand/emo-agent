import numpy as np
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer


class VADPredictor:
    """Predicts Valence, Arousal, Dominance scores for text."""

    def __init__(self, artifacts_dir: str = "artifacts_v1"):
        self.artifacts = Path(artifacts_dir)
        self.model = joblib.load(self.artifacts / "vad_regressor.pkl")
        self.scaler = joblib.load(self.artifacts / "vad_scaler.pkl")
        self.encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def predict(self, text: str) -> dict[str, float]:
        """
        Predict VAD scores for a single text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with keys 'valence', 'arousal', 'dominance'
        """
        embedding = self.encoder.encode([text], normalize_embeddings=True)
        embedding = np.asarray(embedding)

        y_scaled = self.model.predict(embedding)
        y_pred = self.scaler.inverse_transform(y_scaled.reshape(1, -1))[0]

        return {
            "valence": float(y_pred[0]),
            "arousal": float(y_pred[1]),
            "dominance": float(y_pred[2])
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """
        Predict VAD scores for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of dictionaries with VAD scores
        """
        embeddings = self.encoder.encode(texts, normalize_embeddings=True, batch_size=64)
        embeddings = np.asarray(embeddings)

        y_scaled = self.model.predict(embeddings)
        y_pred = self.scaler.inverse_transform(y_scaled)

        return [
            {
                "valence": float(row[0]),
                "arousal": float(row[1]),
                "dominance": float(row[2])
            }
            for row in y_pred
        ]


if __name__ == "__main__":
    predictor = VADPredictor()

    # Example usage
    test_texts = [
        "I am so happy and excited!",
        "This is terrible and depressing.",
        "The weather is okay today."
    ]

    print("Single prediction:")
    result = predictor.predict(test_texts[0])
    print(f"  {test_texts[0]}")
    print(f"  V={result['valence']:.3f}, A={result['arousal']:.3f}, D={result['dominance']:.3f}\n")

    print("Batch prediction:")
    results = predictor.predict_batch(test_texts)
    for text, res in zip(test_texts, results):
        print(f"  {text}")
        print(f"  V={res['valence']:.3f}, A={res['arousal']:.3f}, D={res['dominance']:.3f}")
