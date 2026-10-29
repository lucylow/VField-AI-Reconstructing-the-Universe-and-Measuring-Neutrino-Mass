"""Dataset registry, suitability scoring, and data-quality contracts.

The registry describes candidate public resources; it does not download data or imply that
all resources are directly interchangeable. See docs/DATASET_SELECTION_PLAN.md first.
"""

from nufield.data_registry.cards import DatasetCard, load_dataset_card, write_dataset_card
from nufield.data_registry.catalog import DATASETS, DatasetSource, get_dataset
from nufield.data_registry.quality import (
    DatasetQualityReport,
    QualityFinding,
    audit_numeric_dataset,
)
from nufield.data_registry.selector import DatasetRecommendation, recommend_datasets

__all__ = [
    "DATASETS",
    "DatasetSource",
    "get_dataset",
    "DatasetRecommendation",
    "recommend_datasets",
    "DatasetCard",
    "load_dataset_card",
    "write_dataset_card",
    "DatasetQualityReport",
    "QualityFinding",
    "audit_numeric_dataset",
]
