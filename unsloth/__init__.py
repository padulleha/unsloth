# Copyright 2023-present, the Unsloth team.
# Licensed under the Apache License, Version 2.0

"""Unsloth - Fast LLM finetuning with optimized kernels.

This package provides efficient fine-tuning capabilities for large language models
using custom CUDA kernels and memory optimizations.

Personal fork notes:
- Added patch_version() utility for quick version comparison
- Added version_info tuple (similar to sys.version_info) for convenience
- Fixed redundancy: patch_version() and version_info do the same thing;
  patch_version() now just returns version_info directly
"""

__version__ = "2024.12.0"
__author__ = "Unsloth Team"
__license__ = "Apache 2.0"

# Convenient version tuple, similar to how sys.version_info works
version_info = tuple(int(x) for x in __version__.split("."))

from unsloth.models import (
    FastLanguageModel,
    FastMistralModel,
    FastLlamaModel,
)
from unsloth.trainer import (
    UnslothTrainer,
    UnslothTrainingArguments,
)
from unsloth.chat_templates import (
    get_chat_template,
    standardize_sharegpt,
    train_on_responses_only,
)

__all__ = [
    "FastLanguageModel",
    "FastMistralModel",
    "FastLlamaModel",
    "UnslothTrainer",
    "UnslothTrainingArguments",
    "get_chat_template",
    "standardize_sharegpt",
    "train_on_responses_only",
    "version_info",
]

def get_version():
    """Return the current version of unsloth."""
    return __version__

def patch_version():
    """Return the version as a tuple of ints for easy comparison.

    Note: This is equivalent to accessing `version_info` directly.

    Example:
        >>> if patch_version() >= (2024, 12, 0):
        ...     print("New enough version")
    """
    # Reuse version_info instead of re-parsing __version__ every call
    return version_info
