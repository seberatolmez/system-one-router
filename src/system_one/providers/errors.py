"""Typed provider errors shared by all provider adapters."""


class ProviderConfigurationError(RuntimeError):
    """Raised when the provider cannot be called with the current settings."""


class ProviderUnavailableError(RuntimeError):
    """Raised when the provider cannot be reached or returns unusable output."""
