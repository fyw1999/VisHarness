"""Typed failures used by the trajectory-runner control flow."""

from __future__ import annotations


class TrajectoryContextLengthExceededError(RuntimeError):
    """One trajectory cannot continue because its model context is too long."""


class TrajectoryModelRequestTimeoutError(RuntimeError):
    """One trajectory cannot continue because its model API request timed out."""


class TrajectoryVisionEncoderCacheExceededError(RuntimeError):
    """One trajectory cannot continue because a visual input exceeds the encoder cache."""


class TrajectoryPersistenceError(RuntimeError):
    """Checkpoint or SFT trajectory persistence failed."""
