import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from emotion_ssm.train.generation_condition_eval import main
if __name__=='__main__':main()
