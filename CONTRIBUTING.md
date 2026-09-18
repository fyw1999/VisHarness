# Contributing

Thank you for contributing to VisHarness.

1. Create a focused branch from `main`.
2. Keep generated trajectories, datasets, model weights, checkpoints, logs,
   credentials, and machine-specific paths out of commits.
3. Add or update tests for behavioral changes.
4. Run the relevant test subset before opening a pull request:

   ```bash
   pytest visharness/tests/test_trajectory_runner.py \
          visharness/tests/test_sft_pipeline.py
   ```

5. Explain any required model, dataset, or tool-server assumptions in the
   pull request.

By submitting a contribution, you agree that it may be distributed under the
Apache License 2.0 used by this project.
