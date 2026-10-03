HW1 test kit - Dave Hagman

Contents (all three go in the same folder):
    team_code/test_agent.py   the agent (entry point: TestAgent)
    team_code/network.py      ClassificationNetwork, imported by test_agent.py
    team_code/model.pth       trained weights (state_dict) loaded by test_agent.py

To run:
    1. Copy the three files into leaderboard/team_code/ of the performance
       benchmark (replacing the provided test_agent.py).
    2. Run the usual leaderboard/scripts/run_evaluation.sh. No extra settings
       are needed: with TEAM_CONFIG unset, test_agent.py loads model.pth from
       its own folder. To use weights stored elsewhere, set TEAM_CONFIG to
       that .pth path.

Requirements are the same as the provided test_agent.py (torch, torchvision,
numpy, opencv-python, Pillow). It runs on a CUDA GPU if one is available,
otherwise on the CPU.
