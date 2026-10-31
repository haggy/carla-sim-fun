#!/bin/bash
# Build the HW2 competition kit, following the HW1 kit the TA verified on 2026-10-02.
#
#   module/hw2/testkit/build_testkit.sh [model.pth] [output.zip]
#
#   model   network written by imitation/train.py
#           (default: module/data/hw2_imitation/models/bev_control_weather.pth)
#   output  zip to write (default: <carla root>/dhagman_hw2_testkit.zip)
#
# Zip layout - the TA copies team_code/* into leaderboard/team_code/:
#   README.txt
#   team_code/test_agent.py
#   team_code/hw2_pipeline/*.py      (the real files; in the repo they are symlinks into module/hw2)
#   team_code/hw2_pipeline/model.pth (legacy torch format - see DrivingPolicy.save)
#
# The model is re-saved in torch's legacy format because the TA most likely runs CARLA 0.9.10 / Python 3.7, where
# torch is at most 1.13; for the same reason every shipped .py file is checked against the Python 3.7 grammar and
# must postpone its annotations.
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$KIT_DIR/../../.." && pwd)"
TEAM_CODE="$ROOT/module/hw2/performance_benchmark/leaderboard/team_code"
PYTHON="${PYTHON:-/home/dave/anaconda3/envs/carla_v1/bin/python}"
PIPELINE_FILES=(common.py network.py policy.py lane_detection.py lateral_control.py longitudinal_control.py
                waypoint_prediction.py)

MODEL="$(realpath "${1:-$ROOT/module/data/hw2_imitation/models/bev_control_weather.pth}")"
OUTPUT="$(realpath -m "${2:-$ROOT/dhagman_hw2_testkit.zip}")"

echo "Exporting $MODEL"
# Also refreshes the repo's team_code/hw2_pipeline/model.pth, so the local benchmark runs exactly what the TA receives.
(cd "$TEAM_CODE/hw2_pipeline" && "$PYTHON" -c "
import sys
import torch
from network import DrivingPolicy
model, input_type = DrivingPolicy.load(sys.argv[1], torch.device('cpu'))
model.save(sys.argv[2], input_type, legacy_format=True)
" "$MODEL" "$TEAM_CODE/hw2_pipeline/model.pth")

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/team_code/hw2_pipeline"
cp "$KIT_DIR/README.txt" "$STAGE/"
cp "$TEAM_CODE/test_agent.py" "$STAGE/team_code/"
# -L dereferences the symlinks so the zip holds the real files
for f in "${PIPELINE_FILES[@]}" model.pth; do
	cp -L "$TEAM_CODE/hw2_pipeline/$f" "$STAGE/team_code/hw2_pipeline/"
done

rm -f "$OUTPUT"
(cd "$STAGE" && zip -q -X -r "$OUTPUT" README.txt team_code)

unzip -tq "$OUTPUT"
for f in "${PIPELINE_FILES[@]}"; do
	unzip -p "$OUTPUT" "team_code/hw2_pipeline/$f" | cmp - "$(realpath "$TEAM_CODE/hw2_pipeline/$f")"
done
"$PYTHON" -c "
import ast, sys
for f in sys.argv[1:]:
    tree = ast.parse(open(f).read(), feature_version=(3, 7))
    future = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == '__future__'
              and any(a.name == 'annotations' for a in n.names)]
    if not future:
        sys.exit(f + ': missing from __future__ import annotations (needed on Python 3.7)')
" "$STAGE/team_code/test_agent.py" "${PIPELINE_FILES[@]/#/$STAGE/team_code/hw2_pipeline/}"
echo "Python 3.7 grammar check passed"

unzip -l "$OUTPUT"
echo "Built $OUTPUT"
