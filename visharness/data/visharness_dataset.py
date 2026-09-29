"""VisHarness dataset integration for verl."""

from __future__ import annotations

import copy
import hashlib
import logging
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

from visharness.prompts import TRAIN_TEST_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def replace_system_prompt(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Return a copied conversation containing exactly one current system prompt."""

    if not isinstance(messages, list):
        raise TypeError(f"prompt messages must be a list, got {type(messages).__name__}")

    system_indices: list[int] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(
                f"prompt message at index {index} must be a dict, "
                f"got {type(message).__name__}"
            )
        if message.get("role") == "system":
            system_indices.append(index)

    if len(system_indices) > 1:
        raise ValueError(
            "prompt must contain at most one system message before dynamic "
            f"replacement, found {len(system_indices)}"
        )

    updated_messages = copy.deepcopy(messages)
    if system_indices:
        updated_messages[system_indices[0]]["content"] = system_prompt
    else:
        updated_messages.insert(0, {"role": "system", "content": system_prompt})
    return updated_messages


def _replace_prompt_in_example(
    example: dict[str, Any],
    *,
    prompt_key: str,
    system_prompt: str,
) -> dict[str, Any]:
    if prompt_key not in example:
        raise KeyError(f"dataset row is missing prompt key {prompt_key!r}")
    return {
        prompt_key: replace_system_prompt(
            example[prompt_key],
            system_prompt,
        )
    }


class VisHarnessDataset(RLHFDataset):
    """Use the current RL prompt with verl's multimodal AgentLoop data path."""

    def maybe_filter_out_long_prompts(self, dataframe=None):
        """Inject the runtime system prompt before verl measures prompt length."""

        dynamic_system_prompt = bool(self.config.get("dynamic_system_prompt", True))
        if dynamic_system_prompt:
            system_prompt = TRAIN_TEST_SYSTEM_PROMPT.strip()
            self.system_prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
            logger.info(
                "Injecting canonical VisHarness system prompt into dataset: sha256=%s",
                self.system_prompt_sha256,
            )
            dataframe = dataframe.map(
                _replace_prompt_in_example,
                fn_kwargs={
                    "prompt_key": self.prompt_key,
                    "system_prompt": system_prompt,
                },
                desc="Injecting current VisHarness system prompt",
            )
        else:
            self.system_prompt_sha256 = None

        return super().maybe_filter_out_long_prompts(dataframe)
