HW2 competition kit - Dave Hagman

Contents:
    team_code/test_agent.py              the agent (entry point: TestAgent)
    team_code/hw2_pipeline/*.py          the HW2 modules test_agent.py imports
    team_code/hw2_pipeline/model.pth     trained policy (bev_control_weather), loaded by test_agent.py

To run:
    1. Copy everything in team_code/ into leaderboard/team_code/ of the performance
       benchmark (replacing the provided test_agent.py; hw2_pipeline/ stays a
       subfolder).
    2. Run the usual leaderboard/scripts/run_evaluation.sh. No extra settings
       are needed: with TEAM_CONFIG unset, test_agent.py loads
       hw2_pipeline/model.pth. To use weights stored elsewhere, set TEAM_CONFIG
       to that .pth path.

Requirements are those of the provided test_agent.py (torch, torchvision,
numpy, opencv-python, Pillow) plus scipy and matplotlib; numpy must be 1.21 or
newer. It runs on a CUDA GPU if one is available, otherwise on the CPU.
