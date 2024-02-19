<!-- # Constr_FL


There are three problems in this repo
- Linear constrained quadratic programming (LCQP)
  
  entry file: lcqp.py
- Nayman Pearson classification
  
  entry file: NPclf.py

  download dataset here: https://drive.google.com/drive/folders/1-7MYuNNOBvOJ_s9-h_lwB14IajkoNCMK?usp=drive_link
  
   and place under the project folder -->
## FedNLP

Git repo for paper "Federated Learning with Convex Global and Local Con-
straints"



## Overview
| task      | datasets|
| ----------- |----------|
| Linear Equality Constrainted Quadratic Programming (LCQP)      | synthetic data (automatically generated when running the scripts) |
| Nayman-Pearson Classification (NP)      | breast-cancer-wisc; adult-a, monks-1 [download :link:](https://drive.
google.com/drive/folders/1-7MYuNNOBvOJ_s9-h_lwB14IajkoNCMK?usp=drive_link)  |
| Classification with Fairness Constraints (Fairness)      | breast-cancer-wisc; adult-a, monks-1 [download :link:](https://drive.google.com/drive/folders/1-7MYuNNOBvOJ_s9-h_lwB14IajkoNCMK?usp=drive_link)  |

## Installation
```bash
git clone https://github.com/PL97/Constr_FL.git
```

## Datasets
Download the dataset using the link in the table. Create a new folder named $data/$ at root and put the downloaded data into it.


## Usage
### Linear Equality Constrainted Quadratic Programming (LCQP) 

```bash
cd LCQP

python lcqp.py
```

### Nayman-Pearson Classification (NP)  
**centralized training** :point_down:
```bash
cd NP/decentralized_alg/

python np.py --dataset [choose from "breast-cancer-wisc", "monks-1", "adult"] \
  --n_clinet [e.g., 1, 5, 10, 20]
```

### Classification with Fairness Constraints (Fairness)
**centralized training** :point_down:
```bash
cd NP/Fairness/wglobal/

python fairClassification.py --n_client [e.g., 1, 5, 10, 20] --repeat_idx [random seed, e.g., 0, 1, 2]
```

## How to cite this work

```bibtex
@misc{he2023proximal,
      title={A proximal augmented Lagrangian based algorithm for federated learning with global and local convex conic constraints}, 
      author={Chuan He and Le Peng and Ju Sun},
      year={2023},
      eprint={2310.10117},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```