"""AI architectures for high-dimensional field-level inference.

Architectures are candidate components, not validated cosmological inference methods. Compare to
power-spectrum, likelihood, and reconstruction baselines before making science claims.
"""

from nufield.models.hybrid_encoder import (
    HybridObservationEncoder,
    PackedObservations,
    pack_observations,
)
from nufield.models.model_cards import MODEL_CARDS, ModelCard, get_model_card, model_registry
from nufield.models.selector import ModelCandidate, recommend_models
from nufield.models.spectral_encoder import FNO3DEmbedding, SpectralConv3d

__all__ = [
    "HybridObservationEncoder",
    "PackedObservations",
    "pack_observations",
    "ModelCandidate",
    "recommend_models",
    "ModelCard",
    "MODEL_CARDS",
    "get_model_card",
    "model_registry",
    "FNO3DEmbedding",
    "SpectralConv3d",
]
