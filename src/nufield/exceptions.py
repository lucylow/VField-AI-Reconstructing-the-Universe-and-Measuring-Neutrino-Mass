"""Domain-specific exceptions used across νField."""


class NuFieldError(Exception):
    """Base exception for expected project-level errors."""


class ConfigurationError(NuFieldError, ValueError):
    """Raised when a configuration violates the declared schema."""


class DataContractError(NuFieldError, ValueError):
    """Raised when dataset shapes or metadata violate the data contract."""


class OptionalDependencyError(NuFieldError, ImportError):
    """Raised when an optional backend is requested but not installed."""


class ScientificValidityError(NuFieldError, RuntimeError):
    """Raised when a caller attempts to use a toy component as validated physics."""
