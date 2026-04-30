# Zola AI Engine: Evolutionary Optimizer (GPU Evolution)

> Developed by [@domedg](https://github.com/domedg) and [@Antonio-Rocchia](https://github.com/Antonio-Rocchia)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/GPU-Accelerated-brightgreen?style=flat-square&logo=pytorch)
![AI](https://img.shields.io/badge/AI-130--Param--Linear--Model-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Tournament--Ready-orange?style=flat-square)

An advanced, high-performance Artificial Intelligence engine for the board game **Zola**. This evolution introduces **GPU-accelerated Genetic Training** and a high-fidelity **130-parameter reinforcement learning model**, capable of deep tree search and real-time evolutionary optimization.

---

## 🧠 Core Architecture & Evolution

The Zola AI Engine has evolved from a 68-parameter heuristic model to a high-dimensional **130-parameter linear evaluator**. 

### 1. Hybrid Deployment Strategy
The project now supports a unique hybrid architecture to ensure both high-performance training and dependency-free deployment:
- **`playerGPU.py`**: High-performance engine utilizing **PyTorch** and **CUDA** for batched state evaluations.
- **`player130params.py`**: A **pure-Python** implementation of the same 130-parameter model. It is completely dependency-free (no `torch`, no `numpy`), making it perfect for submission on any standard academic CPU environment.
- **`playerAdvanced.py`**: The original 68-parameter heuristic baseline ("The Truth").

### 2. NegaMax with Alpha-Beta & Iterative Deepening
The backbone remains a highly optimized NegaMax implementation with:
- **Alpha-Beta Pruning**: Drastic reduction of the search space.
- **Zobrist Hashing**: Incrementally updated 64-bit hashes for instant state recognition.
- **Transposition Tables (TT)**: Global cache to avoid redundant calculations, supporting up to 5M+ nodes.
- **Iterative Deepening**: Ensures the engine always returns a move within the 3.0s limit by progressively searching deeper layers.

---

## 🧬 GPU-Accelerated Genetic Training

The new training pipeline (`train_gpu_genetic.py`) utilizes the raw power of the GPU to simulate an entire evolutionary generation in seconds.

### 1. 130-Parameter Model
The model evaluates board states using:
- **64 Red Positional Weights**: One for each square of the board.
- **64 Blue Positional Weights**: Independent weights for the second player.
- **Red & Blue Clustering Weights**: Learned weights for defensive connectivity.

### 2. Massively Parallel Batching
Instead of evaluating one game at a time, the **Genetic GPU Trainer** batches board evaluations for the entire population. It can simulate **hundreds of games simultaneously** on the GPU, allowing the AI to play through thousands of matches in minutes.

### 3. Evolutionary Pipeline
- **Smart Initialization**: The GA starts by converting the weights of the existing "Advanced" bot into the 130-parameter format, giving the evolution a strong baseline.
- **Round-Robin Sparring**: Individuals in the population compete against each other and against the original `playerAdvanced` bot to prove their superiority.
- **Elitism & Cross-over**: The top performers (the Elite) pass their traits to the next generation through blend cross-over and random mutations.

---

## 📂 Repository Structure

```text
📦 Zola-AI-Engine
 ┣ 📂 logs/                # Training logs and trained model weights
 ┃ ┣ 📜 gpu_weights.pth    # Trained PyTorch model
 ┃ ┣ 📜 best_weights_130.json # JSON weights for pure-Python loading
 ┃ ┗ 📜 genetic_training.log # Evolutionary history
 ┣ 📂 scripts/
 ┃ ┣ 📜 train_gpu_genetic.py # Fast GPU-parallelized Genetic Optimizer
 ┃ ┣ 📜 export_weights.py  # Bridges the gap from GPU to pure-Python
 ┃ ┗ 📜 run_tournament.py  # Analytics dashboard for AI vs AI
 ┣ 📂 src/
 ┃ ┣ 📜 ZolaGameS.py       # Core GUI & Engine Dispatcher
 ┃ ┣ 📜 playerGPU.py       # Pytorch GPU Engine
 ┃ ┣ 📜 player130params.py # Pure-Python 130-param Bot (Dependency-free)
 ┃ ┣ 📜 playerAdvanced.py  # Legacy Heuristic Engine (68-param)
 ┃ ┗ 📜 playerExampleAlpha.py # Baseline Opponent
 ┗ 📜 README.md
```

---

## 🚀 Usage

### 1. Playing the Game
Launch the GUI:
```bash
python src/ZolaGameS.py
```
You can select the engine for each player:
- **Gpu**: The fastest engine (requires `torch`).
- **130params**: High-performance engine (Pure Python).
- **Advanced**: Original 68-parameter engine.

### 2. Training the AI
To start the evolutionary optimization:
```bash
conda run -n zola python scripts/train_gpu_genetic.py
```
Monitor the progress: `tail -f logs/genetic_training.log`

### 3. Exporting Weights
After training, convert the GPU brain to a pure-Python brain:
```bash
conda run -n zola python scripts/export_weights.py
```

---

## ⚙️ Environment Setup

Requirements for training/GPU execution:
- Python 3.8+
- PyTorch (CUDA supported recommended)
- NumPy

*The pure-Python `player130params.py` requires **zero** external dependencies and is compatible with any standard Python 3.x environment.*
