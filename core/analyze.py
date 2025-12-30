import numpy as np
import torch
from core.sae import SparseAutoencoder
from core.classifier import DeceptionClassifier
from core.utils import softmax, normalize

# --- load trained components (mocked for v1) ---
sae = SparseAutoencoder()
classifier = DeceptionClassifier()

# NOTE:
# In v1, we simulate activations instead of extracting real ones.
# This keeps the demo reliable and ethical.

def simulate_activation(text):
    """
    Deterministic, biased activation simulation.
    Encodes deception-relevant language cues into the vector.
    """
    rng = np.random.default_rng(abs(hash(text)) % (2**32))

    # base activation
    vec = rng.normal(size=768)

    text_lower = text.lower()

    # --- deception cues ---
    fabrication_cues = [
        "definitely", "guaranteed", "100%", "everyone knows",
        "proven", "no doubt"
    ]

    withholding_cues = [
        "i can’t", "i cannot", "not allowed", "i won’t",
        "policy", "cannot help", "unable to provide"
    ]

    truthful_cues = [
        "it depends", "uncertain", "not sure", "varies",
        "approximately", "estimate"
    ]

    # bias vector based on cues
    for cue in fabrication_cues:
        if cue in text_lower:
            vec[:256] += 2.0   # fabrication region

    for cue in withholding_cues:
        if cue in text_lower:
            vec[256:512] += 2.0  # withholding region

    for cue in truthful_cues:
        if cue in text_lower:
            vec[512:] += 1.5   # truthful region

    return normalize(vec)

def analyze_response(prompt, response):
    """
    Core analysis function.
    Returns a structured deception profile.
    """

    activation = simulate_activation(prompt + response)
    activation_tensor = torch.tensor(activation).float()

    _, features = sae(activation_tensor)
    features = features.detach().numpy()

    probs = classifier.predict_proba(features.reshape(1, -1))

    labels = ["truthful", "fabrication", "withholding"]
    confidence = float(np.max(probs))
    predicted = labels[int(np.argmax(probs))]

    # fake ablation: remove top feature
    ablated = features.copy()
    top_idx = np.argmax(ablated)
    ablated[top_idx] = 0.0

    ablated_probs = classifier.predict_proba(ablated.reshape(1, -1))

    return {
        "prediction": predicted,
        "confidence": round(confidence, 3),
        "probabilities": {
            labels[i]: round(float(probs[i]), 3)
            for i in range(len(labels))
        },
        "top_feature_index": int(top_idx),
        "ablation_effect": round(
            float(confidence - np.max(ablated_probs)), 3
        )
    }
