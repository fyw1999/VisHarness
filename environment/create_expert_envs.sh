#!/usr/bin/env bash

set -euo pipefail

ENVIRONMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not available on PATH." >&2
    exit 1
fi

create_environment() {
    local environment_name="$1"
    local environment_file="$2"

    if conda env list | awk -v name="${environment_name}" '$1 == name { found = 1 } END { exit !found }'; then
        echo "Conda environment ${environment_name} already exists; skipping creation."
        return
    fi

    echo "Creating Conda environment ${environment_name} from ${environment_file}"
    conda env create --file "${environment_file}"
}

repair_super_resolution_environment() {
    echo "Checking BasicSR compatibility in the SuperResolution environment..."
    conda run --no-capture-output -n SuperResolution python -c '
from importlib.metadata import distribution
from pathlib import Path
import os
import shutil
import tempfile

# Locate the package without importing BasicSR before its import is repaired.
package = distribution("basicsr")
if package.version != "1.4.2":
    raise RuntimeError(f"Expected basicsr==1.4.2, found {package.version}; refusing to modify an unexpected version.")

source = Path(package.locate_file("basicsr/data/degradations.py")).resolve()
content = source.read_bytes()
lines = content.decode("utf-8").splitlines(keepends=True)
old_import = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
new_import = "from torchvision.transforms.functional import rgb_to_grayscale"
old_lines = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == old_import]
new_lines = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == new_import]

if len(old_lines) == 1 and not new_lines:
    # Check that the replacement API exists before modifying the dependency.
    from torchvision.transforms.functional import rgb_to_grayscale

    backup = source.with_name(source.name + ".visharness.bak")
    if not backup.exists():
        shutil.copy2(source, backup)
    index = old_lines[0]
    lines[index] = lines[index].replace(old_import, new_import, 1)
    descriptor, temporary = tempfile.mkstemp(prefix=".visharness-basicsr-", dir=source.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write("".join(lines).encode("utf-8"))
        shutil.copymode(source, temporary)
        os.replace(temporary, source)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Applied BasicSR TorchVision import fix: {source}; original backup: {backup}", flush=True)
elif not old_lines and len(new_lines) == 1:
    print(f"BasicSR TorchVision import is already compatible: {source}", flush=True)
else:
    raise RuntimeError(f"Unexpected BasicSR import layout in {source}; refusing to modify it.")

# These imports initialize no model weights and must pass before reporting ready.
import basicsr
import gfpgan
import realesrgan

print("SuperResolution imports verified: basicsr, gfpgan, realesrgan.", flush=True)
'
}

create_environment "Molmo" "${ENVIRONMENT_DIR}/experts/Molmo.yml"
create_environment "SAM" "${ENVIRONMENT_DIR}/experts/SAM.yml"
create_environment "SuperResolution" "${ENVIRONMENT_DIR}/experts/SuperResolution.yml"
# Run even for existing environments, which may still have the original import.
repair_super_resolution_environment

echo "Visual expert environments are ready."
