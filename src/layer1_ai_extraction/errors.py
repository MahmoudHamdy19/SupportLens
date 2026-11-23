"""Exceptions raised by Layer 1."""


class ModelUnavailableError(RuntimeError):
    """A required model, adapter, library or device is missing.

    Raised in strict mode instead of silently degrading to another extractor.
    """


class ExtractionFormatError(ValueError):
    """Generated text could not be turned into a valid commitment tuple."""
