#!/bin/bash
# Build the TA test kit: the layout the TA verified on 2026-10-02.
#
#   module/hw1/testkit/build_testkit.sh [checkpoint.pth] [output.zip]
#
#   checkpoint  pickled model written by training.py
#               (default: module/data/dropout/model.pth)
#   output      zip to write (default: <carla root>/dhagman_hw1_testkit.zip)
#
# Zip layout - the TA copies team_code/* into leaderboard/team_code/:
#   README.txt
#   team_code/test_agent.py
#   team_code/network.py   (the real file; team_code/network.py in the repo is a symlink)
#   team_code/model.pth    (state_dict only, legacy format - see export_weights)
#
# The checkpoint is re-exported as weights only because the pickled model is
# tied to the module.hw1.network import path and to our torch version, neither
# of which exist on the TA's CARLA 0.9.10 / Python 3.7 setup.
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$KIT_DIR/../../.." && pwd)"
TEAM_CODE="$ROOT/module/hw1/performance_benchmark/leaderboard/team_code"

CHECKPOINT="$(realpath "${1:-$ROOT/module/data/dropout/model.pth}")"
OUTPUT="$(realpath -m "${2:-$ROOT/dhagman_hw1_testkit.zip}")"

echo "Exporting weights from $CHECKPOINT"
# Also refreshes the repo's team_code/model.pth, so the local benchmark runs
# exactly what the TA receives.
(cd "$ROOT" && python3 -c "
import sys
from module.hw1.network import ClassificationNetwork
ClassificationNetwork.load_and_eval(sys.argv[1]).export_weights(sys.argv[2])
" "$CHECKPOINT" "$TEAM_CODE/model.pth")

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir "$STAGE/team_code"
cp "$KIT_DIR/README.txt" "$STAGE/"
# -L dereferences the network.py symlink so the zip holds the real file
cp -L "$TEAM_CODE/test_agent.py" "$TEAM_CODE/network.py" "$TEAM_CODE/model.pth" "$STAGE/team_code/"

rm -f "$OUTPUT"
(cd "$STAGE" && zip -q -X -r "$OUTPUT" README.txt team_code)

unzip -tq "$OUTPUT"
unzip -p "$OUTPUT" team_code/network.py | cmp - "$ROOT/module/hw1/network.py"
python3 -c "
import ast, sys
for f in sys.argv[1:]:
    ast.parse(open(f).read(), feature_version=(3, 7))
" "$ROOT/module/hw1/network.py" "$TEAM_CODE/test_agent.py"
echo "Python 3.7 grammar check passed"

unzip -l "$OUTPUT"
echo "Built $OUTPUT"
