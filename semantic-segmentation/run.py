#!/usr/bin/env python
"""Run one federated segmentation experiment.

Examples
  python run.py --dataset camvid --algorithm fedavg  --partition natural --num_clients 3
  python run.py --dataset camvid --algorithm fedadmm --rho 0.01
  python run.py --dataset camvid --algorithm fedavg  --bn_mode fedbn --eval_personalized true
  python run.py --dataset pets   --algorithm fedprox --mu 0.01 --partition dirichlet --dirichlet_alpha 0.3 --num_clients 8
  python run.py --dataset synthetic --model unet --rounds 2 --device cpu      # 30-second smoke test
"""
from fedseg import parse_args, run_experiment

if __name__ == "__main__":
    run_experiment(parse_args())
