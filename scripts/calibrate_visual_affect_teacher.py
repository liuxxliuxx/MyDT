"""CLI for human-labelled, source-separated FLAME adapter calibration."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from emotion_ssm.train.visual_teacher_calibration import main
if __name__=='__main__': main()
