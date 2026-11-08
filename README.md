# Federated learning

Two projects on federated optimization. In both, the data stays split across clients and only model updates
are shared.

| | task | data | methods |
|---|---|---|---|
| [`shakespeare-next-char`](shakespeare-next-char) | next-character prediction (LSTM) | Shakespeare, 10 clients, IID and non-IID | FedAvg, FedProx, LiADMM, FedADMM |
| [`semantic-segmentation`](semantic-segmentation) | semantic segmentation (DeepLabV3, SegNet) | CamVid (one client per car), Oxford-IIIT Pet (cat-only / dog-only clients) | FedAvg, FedProx, FedADMM, FedDyn, SCAFFOLD, FedNova, FedBN / SiloBN |

Each folder has its own README with the results.
