"""fedseg - federated semantic segmentation (FedAvg / FedProx / FedADMM / FedDyn / SCAFFOLD / FedNova,
with FedBN / SiloBN and FedAvgM / FedAdam server optimisers)."""
from .config import Config, parse_args
from .trainer import FederatedTrainer, run_experiment

__all__ = ["Config", "parse_args", "FederatedTrainer", "run_experiment"]
__version__ = "2.0.0"
