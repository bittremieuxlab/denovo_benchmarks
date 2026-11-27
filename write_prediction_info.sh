#!/bin/bash
# Write prediction_info.yaml next to output.csv: when, from which commit of this repository and with which
# container image the predictions were made. evaluate.py copies it into results/provenance.csv.
# Usage: write_prediction_info.sh algorithm_name output_dir
algorithm_name="$1"
output_dir="$2"

commit=$(git rev-parse HEAD 2>/dev/null)
if [ -n "$commit" ] && [ -n "$(git status --porcelain --untracked-files=no -- "algorithms/${algorithm_name}")" ]; then
    commit="${commit}-dirty"
fi
cat > "${output_dir}/prediction_info.yaml" << EOF
predicted: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit: "${commit}"
image_sha256: "$(sha256sum "algorithms/${algorithm_name}/container.sif" | cut -d " " -f 1)"
EOF
