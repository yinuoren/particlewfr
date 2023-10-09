# ParticleMethodsForMultiObjectiveOptimization

This repository include scripts to reproduce experiments for profiling the Pareto front in Multi-Objective Optimization (MOO).

The implementation is based on the official implementation for COSMOS: a method to learn Pareto fronts that scales to large datasets and deep models.

For details see paper.

## Usage

1. Download the dataset as described in `readme.md` in the respective data folder.
1. Run the code:

```bash
python multi_objective/main.py --dataset mslr --method particle
```

For the logs and results see newly created folder `results`.

For the settings see [settings.py](multi_objective/settings.py)

---

Available datasets:

| command-line option  | Description                  |
|----------------------|------------------------------|
| `-d mslr`            | MSLR-WEB10K dataset          |
| `-d zdt1`            | ZDT1 Problem                 |
| `-d zdt2`            | ZDT2 Problem                 |
| `-d zdt3`            | ZDT3 Problem                 |
| `-d dtlz7`           | DTLZ7 Problem                |

---

Available algorithms:

| command-line option  | Description                                         |
|----------------------|-----------------------------------------------------|
| `-m cosmos`          | COSMOS algorithm [1]                                | 
| `-m hyper_ln`        | PHN (Linear Scalarization) algorithm [2]            | 
| `-m hyper_epo`       | PHN (EPO) algorithm [2]                             | 
| `-m pmtl`            | ParetoMTL algorithm [3]                             | 
| `-m single_task`     | Treat each objective as single task                 | 
| `-m uniform`         | Uniform scaling of all objectives                   | 
| `-m argmo_hv`        | ARGMO wrt hypervolume [4]                           |
| `-m argmo_kernel`    | ARGMO with kernel [4]                               |
| `-m particle`        | Particle methods with Wasserstein-Fisher-Rao flow   |


[1] Ruchte, Michael, and Josif Grabocka. "Scalable pareto front approximation for deep multi-objective learning." 2021 IEEE international conference on data mining (ICDM). IEEE, 2021.

[2] Navon, Aviv, et al. "Learning the Pareto Front with Hypernetworks." International Conference on Learning Representations. 2020.

[3] Lin, Xi, et al. "Pareto multi-task learning." Advances in neural information processing systems 32 (2019).

[4] Chen, Weiyu, and James Kwok. "Multi-Objective Deep Learning with Adaptive Reference Vectors." Advances in Neural Information Processing Systems 35 (2022): 32723-32735.

[5] Liu, Xingchao, Xin Tong, and Qiang Liu. "Profiling pareto front with multi-objective stein variational gradient descent." Advances in Neural Information Processing Systems 34 (2021): 14721-14733.


## Installation

Requirements:
1. CUDA capable GPU

Create a venv:

```bash
python3 -m venv moo
source moo/bin/activate
```

Clone repository:

```
git clone ...
cd moo
```

Install requirements:

```
pip install -r requirements.txt
```

The large number of dependencies is partly due to the baselines, available in this repository as well.

## Acknowledgments

We thank the authors of [1] for creating a helpful code framework to compare different algorithms. We are also thankful to the authors of [4, 5] for sharing their code, contributing significantly to our research.

