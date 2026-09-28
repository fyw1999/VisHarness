"""VisHarness dataset integration for verl."""

from verl.utils.dataset.rl_dataset import RLHFDataset


class VisHarnessDataset(RLHFDataset):
    """Use verl's native multimodal AgentLoop data path for VisHarness samples."""

