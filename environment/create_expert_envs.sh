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
        echo "Conda environment ${environment_name} already exists; leaving it unchanged."
        return
    fi

    echo "Creating Conda environment ${environment_name} from ${environment_file}"
    conda env create --file "${environment_file}"
}

create_environment "Molmo" "${ENVIRONMENT_DIR}/experts/Molmo.yml"
create_environment "SAM" "${ENVIRONMENT_DIR}/experts/SAM.yml"
create_environment "SuperResolution" "${ENVIRONMENT_DIR}/experts/SuperResolution.yml"

echo "Visual expert environments are ready."
