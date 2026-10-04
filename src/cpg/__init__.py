"""Experimental graph frontend. Reference execution must be explicit."""
from .pipeline import Pipeline
from .config import ConfigError, load_pipeline

__all__ = ["Pipeline", "ConfigError", "load_pipeline"]
