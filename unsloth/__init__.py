# Copyright 2023-present, the Unsloth team.
# Licensed under the Apache License, Version 2.0

"""Unsloth - Fast LLM finetuning with optimized kernels.

This package provides efficient fine-tuning capabilities for large language models
using custom CUDA kernels and memory optimizations.
"""

__version__ = "2024.12.0"
__author__ = "Unsloth Team"
__license__ = "Apache 2.0"

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
]

def get_version():
    """Return the current version of unsloth."""
    return __version__
