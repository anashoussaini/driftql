<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Drift Q-Learning</h1>
<div style="margin: 15px 0;">
  <img
    src="https://img.shields.io/badge/Accepted%20to-NeurIPS%202026-e97822"
    alt="Accepted to NeurIPS 2026"
  >
</div>
      <h2>
        <a href="https://arxiv.org/abs/2606.00350">Paper</a>
        &emsp;
        <a href="https://driftql.github.io/">Project page</a>
      </h2>
    </summary>
  </ul>
</div>

<img src="assets/driftql.gif" width="90%" alt="Drift Q-Learning Animation">

</div>

## Overview

Drift Q-learning (DriftQL) is a simple and performant data-driven RL algorithm that leverages an expressive policy to model complex action distributions in data.

> **Note:** DriftQL's codebase is based on [FQL's implementation](https://github.com/seohongpark/fql/), with Diffusion-QL and Implicit Diffusion-QL (IDQL) added, both based on official author implementations.


## Installation

All packages are based on FQL's codebase, and the installation process is the same as FQL's. For convenience, we provide the installation instructions here again.

The current project requires `Python 3.10+` and is based on JAX. The main dependencies are `jax >= 0.6.2`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`. To install the full dependencies, simply run:
```bash
pip install -r requirements.txt
```
To use D4RL environments, you need to additionally set up MuJoCo 2.1.0. `mujoco-py` expects the library at `~/.mujoco/mujoco210`:
```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
tar -xzf /tmp/mujoco210.tar.gz -C ~/.mujoco
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc
```

Alternatively, you can set up an isolated environment with [Mamba](https://mamba.readthedocs.io/) (or Conda):
```bash
mamba create -n driftql python=3.10 -y
mamba activate driftql
pip install -r requirements.txt
```

To ensure that Jax is installed correctly with GPU support, run the following:
```bash
python jax_check.py 
```

By default, `pip install jax` only installs the CPU wheel. If `jax_check.py` reports a `CpuDevice` instead of `CudaDevice`, install the CUDA-bundled wheels (requires an NVIDIA driver supporting CUDA 12):
```bash
pip install -U "jax[cuda12]"
```

If you see the following output, then Jax is successfully installed and can access the GPU:
```
JAX version: 0.6.2
Devices: [CudaDevice(id=0)]

1. Creating 10000x10000 matrices...
✓ Arrays loaded to GPU.

2. Compiling (JIT)...
✓ Compiled.

3. Stressing GPU (check nvidia-smi now!)...
✓ Completed 50 heavy operations in 0.95 seconds.
```


## Usage

The main implementation of DriftQL is in [agents/driftql.py](agents/driftql.py), and our implementations of baselines can also be found in the same directory.
Here are some example commands:

```bash
# DriftQL on OGBench antsoccer-arena-task1
python main.py --env_name=antsoccer-arena-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.alpha=10 --agent.drift_temp=0.5 --agent.q_agg=mean

# DriftQL on D4RL halfcheetah-medium-expert
python main.py --env_name=halfcheetah-medium-expert-v2 --agent.drift_temp=0.5 --agent.alpha=300 --agent.q_agg=mean
```

🌟 The full set of commands used to produce our results is listed below. The `--agent.*` flags override
entries in `get_config()`; any hyperparameter not shown keeps its default (e.g. `lr=3e-4`,
`batch_size=256`, `tau=0.005`, `kernel=laplace`, `q_agg=min`, `discount=0.99`, `drift_ngen=32`,
`noise_dim=action_dim`). Add `--seed=<n>` to change the random seed. For the OGBench suites we report
over five singletask variants (`task1`–`task5`), all listed explicitly.

<details>
<summary><b>OGBench</b></summary>

```bash
# humanoidmaze-medium-navigate
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent.drift_temp=0.5 --agent.alpha=65 --agent.discount=0.998 --agent.q_agg=mean
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --agent.drift_temp=0.5 --agent.alpha=65 --agent.discount=0.998 --agent.q_agg=mean
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 --agent.drift_temp=0.5 --agent.alpha=65 --agent.discount=0.998 --agent.q_agg=mean
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 --agent.drift_temp=0.5 --agent.alpha=65 --agent.discount=0.998 --agent.q_agg=mean
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task5-v0 --agent.drift_temp=0.5 --agent.alpha=65 --agent.discount=0.998 --agent.q_agg=mean

# humanoidmaze-large-navigate
python main.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --agent.drift_temp=0.2 --agent.alpha=32 --agent.discount=0.995 --agent.q_agg=mean
python main.py --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --agent.drift_temp=0.2 --agent.alpha=32 --agent.discount=0.995 --agent.q_agg=mean
python main.py --env_name=humanoidmaze-large-navigate-singletask-task3-v0 --agent.drift_temp=0.2 --agent.alpha=32 --agent.discount=0.995 --agent.q_agg=mean
python main.py --env_name=humanoidmaze-large-navigate-singletask-task4-v0 --agent.drift_temp=0.2 --agent.alpha=32 --agent.discount=0.995 --agent.q_agg=mean
python main.py --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --agent.drift_temp=0.2 --agent.alpha=32 --agent.discount=0.995 --agent.q_agg=mean

# antmaze-large-navigate
python main.py --env_name=antmaze-large-navigate-singletask-task1-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.q_agg=mean
python main.py --env_name=antmaze-large-navigate-singletask-task2-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.q_agg=mean
python main.py --env_name=antmaze-large-navigate-singletask-task3-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.q_agg=mean
python main.py --env_name=antmaze-large-navigate-singletask-task4-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.q_agg=mean
python main.py --env_name=antmaze-large-navigate-singletask-task5-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.q_agg=mean

# antmaze-giant-navigate
python main.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --agent.drift_temp=0.2 --agent.alpha=10 --agent.discount=0.995
python main.py --env_name=antmaze-giant-navigate-singletask-task2-v0 --agent.drift_temp=0.2 --agent.alpha=10 --agent.discount=0.995
python main.py --env_name=antmaze-giant-navigate-singletask-task3-v0 --agent.drift_temp=0.2 --agent.alpha=10 --agent.discount=0.995
python main.py --env_name=antmaze-giant-navigate-singletask-task4-v0 --agent.drift_temp=0.2 --agent.alpha=10 --agent.discount=0.995
python main.py --env_name=antmaze-giant-navigate-singletask-task5-v0 --agent.drift_temp=0.2 --agent.alpha=10 --agent.discount=0.995

# antsoccer-arena-navigate
python main.py --env_name=antsoccer-arena-navigate-singletask-task1-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.discount=0.995 --agent.q_agg=mean
python main.py --env_name=antsoccer-arena-navigate-singletask-task2-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.discount=0.995 --agent.q_agg=mean
python main.py --env_name=antsoccer-arena-navigate-singletask-task3-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.discount=0.995 --agent.q_agg=mean
python main.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.discount=0.995 --agent.q_agg=mean
python main.py --env_name=antsoccer-arena-navigate-singletask-task5-v0 --agent.drift_temp=0.5 --agent.alpha=10 --agent.discount=0.995 --agent.q_agg=mean

# cube-single-play
python main.py --env_name=cube-single-play-singletask-task1-v0 --agent.drift_temp=0.02 --agent.alpha=60 --agent.discount=0.995 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=cube-single-play-singletask-task2-v0 --agent.drift_temp=0.02 --agent.alpha=60 --agent.discount=0.995 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=cube-single-play-singletask-task3-v0 --agent.drift_temp=0.02 --agent.alpha=60 --agent.discount=0.995 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=cube-single-play-singletask-task4-v0 --agent.drift_temp=0.02 --agent.alpha=60 --agent.discount=0.995 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=cube-single-play-singletask-task5-v0 --agent.drift_temp=0.02 --agent.alpha=60 --agent.discount=0.995 --agent.q_agg=mean --agent.kernel=gaussian

# cube-double-play
python main.py --env_name=cube-double-play-singletask-task1-v0 --agent.drift_temp=0.2 --agent.drift_ngen=4 --agent.noise_dim=8 --agent.alpha=100 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=cube-double-play-singletask-task2-v0 --agent.drift_temp=0.2 --agent.drift_ngen=4 --agent.noise_dim=8 --agent.alpha=100 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=cube-double-play-singletask-task3-v0 --agent.drift_temp=0.2 --agent.drift_ngen=4 --agent.noise_dim=8 --agent.alpha=100 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=cube-double-play-singletask-task4-v0 --agent.drift_temp=0.2 --agent.drift_ngen=4 --agent.noise_dim=8 --agent.alpha=100 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=cube-double-play-singletask-task5-v0 --agent.drift_temp=0.2 --agent.drift_ngen=4 --agent.noise_dim=8 --agent.alpha=100 --agent.q_agg=mean --agent.kernel=gaussian

# scene-play
python main.py --env_name=scene-play-singletask-task1-v0 --agent.drift_temp=0.2 --agent.alpha=250 --agent.discount=0.995 --agent.kernel=gaussian
python main.py --env_name=scene-play-singletask-task2-v0 --agent.drift_temp=0.2 --agent.alpha=250 --agent.discount=0.995 --agent.kernel=gaussian
python main.py --env_name=scene-play-singletask-task3-v0 --agent.drift_temp=0.2 --agent.alpha=250 --agent.discount=0.995 --agent.kernel=gaussian
python main.py --env_name=scene-play-singletask-task4-v0 --agent.drift_temp=0.2 --agent.alpha=250 --agent.discount=0.995 --agent.kernel=gaussian
python main.py --env_name=scene-play-singletask-task5-v0 --agent.drift_temp=0.2 --agent.alpha=250 --agent.discount=0.995 --agent.kernel=gaussian

# puzzle-3x3-play
python main.py --env_name=puzzle-3x3-play-singletask-task1-v0 --agent.drift_temp=0.5 --agent.alpha=50 --agent.kernel=gaussian
python main.py --env_name=puzzle-3x3-play-singletask-task2-v0 --agent.drift_temp=0.5 --agent.alpha=50 --agent.kernel=gaussian
python main.py --env_name=puzzle-3x3-play-singletask-task3-v0 --agent.drift_temp=0.5 --agent.alpha=50 --agent.kernel=gaussian
python main.py --env_name=puzzle-3x3-play-singletask-task4-v0 --agent.drift_temp=0.5 --agent.alpha=50 --agent.kernel=gaussian
python main.py --env_name=puzzle-3x3-play-singletask-task5-v0 --agent.drift_temp=0.5 --agent.alpha=50 --agent.kernel=gaussian

# puzzle-4x4-play
python main.py --env_name=puzzle-4x4-play-singletask-task1-v0 --agent.drift_temp=0.8 --agent.alpha=300 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=puzzle-4x4-play-singletask-task2-v0 --agent.drift_temp=0.8 --agent.alpha=300 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=puzzle-4x4-play-singletask-task3-v0 --agent.drift_temp=0.8 --agent.alpha=300 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=puzzle-4x4-play-singletask-task4-v0 --agent.drift_temp=0.8 --agent.alpha=300 --agent.q_agg=mean --agent.kernel=gaussian
python main.py --env_name=puzzle-4x4-play-singletask-task5-v0 --agent.drift_temp=0.8 --agent.alpha=300 --agent.q_agg=mean --agent.kernel=gaussian
```

</details>

<details>
<summary><b>D4RL - AntMaze</b></summary>

```bash
# antmaze-large-play
python main.py --env_name=antmaze-large-play-v2 --agent.drift_temp=0.5 --agent.alpha=3 --agent.discount=0.995 --agent.kernel=gaussian

# antmaze-medium-diverse
python main.py --env_name=antmaze-medium-diverse-v2 --agent.drift_temp=0.5 --agent.alpha=8 --agent.discount=0.995 --agent.kernel=gaussian

# antmaze-medium-play
python main.py --env_name=antmaze-medium-play-v2 --agent.drift_temp=0.5 --agent.alpha=5 --agent.kernel=gaussian

# antmaze-umaze-diverse
python main.py --env_name=antmaze-umaze-diverse-v2 --agent.drift_temp=0.5 --agent.alpha=12 --agent.q_agg=mean

# antmaze-umaze
python main.py --env_name=antmaze-umaze-v2 --agent.drift_temp=0.5 --agent.alpha=15

# antmaze-large-diverse
python main.py --env_name=antmaze-large-diverse-v2 --agent.drift_temp=0.5 --agent.alpha=5 --agent.discount=0.995 --agent.kernel=gaussian
```

</details>

<details>
<summary><b>D4RL - Adroit</b></summary>

```bash
# door-cloned
python main.py --env_name=door-cloned-v1 --agent.drift_temp=0.2 --agent.alpha=4500 --agent.discount=0.995

# door-expert
python main.py --env_name=door-expert-v1 --agent.drift_temp=0.2 --agent.alpha=4500 --agent.discount=0.995 --agent.q_agg=mean

# door-human
python main.py --env_name=door-human-v1 --agent.drift_temp=0.2 --agent.alpha=4500 --agent.discount=0.995 --agent.q_agg=mean

# hammer-cloned
python main.py --env_name=hammer-cloned-v1 --agent.drift_temp=0.05 --agent.alpha=2500 --agent.q_agg=mean

# hammer-expert
python main.py --env_name=hammer-expert-v1 --agent.drift_temp=0.05 --agent.alpha=2500 --agent.discount=0.995

# hammer-human
python main.py --env_name=hammer-human-v1 --agent.drift_temp=0.05 --agent.alpha=2500 --agent.discount=0.995 --agent.q_agg=mean

# pen-cloned
python main.py --env_name=pen-cloned-v1 --agent.drift_temp=0.05 --agent.alpha=1500 --agent.discount=0.995 --agent.kernel=gaussian

# pen-expert
python main.py --env_name=pen-expert-v1 --agent.drift_temp=0.9 --agent.alpha=2000

# pen-human
python main.py --env_name=pen-human-v1 --agent.drift_temp=0.05 --agent.alpha=2000 --agent.discount=0.995 --agent.q_agg=mean

# relocate-cloned
python main.py --env_name=relocate-cloned-v1 --agent.drift_temp=0.2 --agent.alpha=5000 --agent.discount=0.995

# relocate-expert
python main.py --env_name=relocate-expert-v1 --agent.drift_temp=0.2 --agent.alpha=5000 --agent.discount=0.995

# relocate-human
python main.py --env_name=relocate-human-v1 --agent.drift_temp=0.2 --agent.alpha=5000 --agent.discount=0.995
```

</details>

<details>
<summary><b>D4RL - Gym locomotion (MuJoCo)</b></summary>

```bash
# halfcheetah-medium-expert
python main.py --env_name=halfcheetah-medium-expert-v2 --agent.drift_temp=0.5 --agent.alpha=300 --agent.q_agg=mean

# halfcheetah-medium-replay
python main.py --env_name=halfcheetah-medium-replay-v2 --agent.drift_temp=0.5 --agent.alpha=10 --agent.discount=0.995

# halfcheetah-medium
python main.py --env_name=halfcheetah-medium-v2 --agent.drift_temp=0.5 --agent.alpha=3 --agent.q_agg=mean

# hopper-medium-expert
python main.py --env_name=hopper-medium-expert-v2 --agent.drift_temp=0.1 --agent.alpha=600 --agent.discount=0.995 --agent.kernel=gaussian

# hopper-medium-replay
python main.py --env_name=hopper-medium-replay-v2 --agent.drift_temp=0.1 --agent.alpha=100 --agent.discount=0.995 --agent.kernel=gaussian

# hopper-medium
python main.py --env_name=hopper-medium-v2 --agent.drift_temp=0.1 --agent.alpha=100 --agent.discount=0.995 --agent.kernel=gaussian

# walker2d-medium-expert
python main.py --env_name=walker2d-medium-expert-v2 --agent.drift_temp=0.1 --agent.alpha=1000 --agent.q_agg=mean

# walker2d-medium-replay
python main.py --env_name=walker2d-medium-replay-v2 --agent.drift_temp=0.1 --agent.alpha=300 --agent.discount=0.995

# walker2d-medium
python main.py --env_name=walker2d-medium-v2 --agent.drift_temp=0.1 --agent.alpha=1000 --agent.discount=0.995
```

</details>


## Citation

If you find this work useful, please cite:

```bibtex
@article{houssaini2026drift,
  title={Drift Q-Learning},
  author={Houssaini, Anas and Danesh, Mohamad H and Abyaneh, Amin and Fujimoto, Scott and Lin, Hsiu-Chin and Meger, David},
  journal={arXiv preprint arXiv:2606.00350},
  year={2026}
}
```
