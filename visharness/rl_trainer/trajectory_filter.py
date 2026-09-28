"""Trajectory-group filtering hooks for VisHarness."""

from verl import DataProto


def filter_trajectory_groups(batch: DataProto) -> DataProto:
    """Filter complete trajectory groups before advantage computation."""

    raise NotImplementedError("Implement VisHarness/T2PO-style trajectory filtering.")
