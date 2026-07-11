"""Train and version a CALO policy checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
from calo_rpd_studio.algorithms.calo.training import TrainingConfig,train_policy

def main():
    p=argparse.ArgumentParser(description="Train the CALO policy controller on the documented synthetic training families.");p.add_argument("--epochs",type=int,default=20);p.add_argument("--episodes",type=int,default=16);p.add_argument("--horizon",type=int,default=24);p.add_argument("--seed",type=int,default=2026);p.add_argument("--output",default=str(Path(__file__).resolve().parents[1]/"data"/"trained_models"/"calo_policy_v1.pt"));a=p.parse_args();cfg=TrainingConfig(a.epochs,a.episodes,a.horizon,a.seed);history=train_policy(cfg,a.output);print(f"Saved CALO policy: {Path(a.output).resolve()}");print(f"Final training record: {history[-1] if history else {}}");return 0
if __name__=="__main__":raise SystemExit(main())
