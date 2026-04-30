# Zola AI Engine: Evolutionary Optimizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![AI](https://img.shields.io/badge/AI-NegaMax%20%2B%20AlphaBeta-red?style=flat-square)
![Genetic Algorithm](https://img.shields.io/badge/Optimization-Genetic%20Algorithm-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Competitive-orange?style=flat-square)

An advanced, high-performance Artificial Intelligence engine for the board game **Zola**. This project aims to push the boundaries of Python's computational limits by utilizing strict **Bitboard representations**, deep tree search algorithms, and a custom **Multi-core Genetic Algorithm** for autonomous heuristic optimization.

---

## 🧠 Core Architecture & Search Algorithms

The Zola AI Engine is designed to maximize computational efficiency, allowing deep state-space exploration within strict temporal constraints (maximum 3 seconds per move).

### 1. NegaMax with Alpha-Beta Pruning
The backbone of the search engine is a highly optimized NegaMax implementation. Instead of writing separate Minimax logic for the maximizing and minimizing players, NegaMax exploits the zero-sum nature of Zola: `max(a, b) == -min(-a, -b)`.
Coupled with Alpha-Beta pruning, the engine drastically reduces the number of nodes evaluated by cutting off branches that are mathematically proven to be suboptimal.

### 2. Iterative Deepening Framework
To guarantee a move is always returned before the strict 3.0s timeout, the engine wraps the NegaMax algorithm in an **Iterative Deepening** loop.
- The engine searches at Depth 1, then Depth 2, then Depth 3, etc.
- A strict timeout check (`timeout - 0.15s` safety margin) is verified at every node.
- If a `TimeoutError` is raised mid-search, the engine catches it and returns the best move found in the *previous* fully completed depth.

### 3. Zobrist Hashing & Massive Transposition Tables
A standard Alpha-Beta search often encounters the exact same board state through different move permutations (transpositions). To avoid recalculating these states:
- A unique 64-bit random integer is assigned to every piece on every square during initialization (**Zobrist Hashing**).
- As moves are made, the hash is updated incrementally using the XOR bitwise operator (`^`), which is extremely fast `(new_hash = old_hash ^ ZOBRIST[from] ^ ZOBRIST[to])`.
- States are stored in a dynamically growing **Transposition Table (TT)**. 
- **Memory Optimization**: The system dynamically monitors RAM usage and automatically clears the cache if the node count exceeds **5,000,000 entries**, ensuring safe execution on systems with 32GB+ RAM.

---

## 🧮 Bitboards & Heuristic Evaluation

Evaluating millions of nodes per second in Python requires eliminating slow loops and object allocations. The board is represented purely by two 64-bit integers (`red_bb` and `blue_bb`).

### The Split-Brain 68-Parameter Genome
Board games are highly asymmetrical. A strategy that works for the First Player often fails for the Second Player. The engine solves this by maintaining a genome of **68 independent parameters**:
- **Indices 0..33**: The evaluation matrix loaded when playing as the First Player (Red). Focuses on initiative and aggressive central control.
- **Indices 34..67**: The evaluation matrix loaded when playing as the Second Player (Blue). Focuses on defensive traps and positional sacrifices.

### Phase-Based Positional Evaluation
The 33 parameters per player are divided into specific phases of the game. The engine dynamically calculates the game phase using the `bit_count()` of the board and shifts the evaluation offset:
```python
total_pieces = red_count + blue_count
if total_pieces > 48:
    phase_offset = 0   # Early Game weights
elif total_pieces > 24:
    phase_offset = 10  # Mid Game weights
else:
    phase_offset = 20  # Late Game weights
```

### Bitwise Clustering Heuristic
To evaluate whether pieces are safely defending each other (clustered) or dangerously isolated, the engine uses raw bit-shifting. By shifting the bitboard horizontally and vertically, we can count adjacent pieces in nanoseconds without nested loops:
```python
# [REDACTED FOR COMPETITIVE INTEGRITY] 
# Example logic demonstrating the bitwise approach:
cluster_score = (board & (board << 1)).bit_count() + \
                (board & (board >> 8)).bit_count()
```

---

## 🧬 Evolutionary Training (Genetic Algorithm)

Instead of manually guessing the perfect values for the 68 parameters, this project features a fully autonomous multi-core training pipeline (`train_zola.py`). The AI evolves its own "DNA" through natural selection.

### 1. Initialization
The population is seeded with arrays of 68 parameters. While one individual is initialized with the baseline `[0, ..., 1000]` material-only strategy, the rest are generated with random integers within logical boundaries (e.g., `-30` to `+40` for positional weights).

### 2. Multi-Processing Fitness Evaluation (Round-Robin)
To evaluate fitness, the entire population plays a brutal Round-Robin tournament.
- The script detects the host machine's physical cores (`multiprocessing.cpu_count() - 1`) and instantiates a `ProcessPoolExecutor`.
- All matches run asynchronously in parallel, isolating memory instances.
- **Scoring**: 3 points for a win, 1 for a draw.

### 3. The Baseline Boss-Fight
To ensure the evolving AI doesn't just overfit to beating its own siblings, every bot in the population is forced to play a "Boss Fight" against the hardcoded `playerExampleAlpha` baseline. 
- Bots must play Alpha both as the First Player and as the Second Player. 
- Defeating the Alpha baseline grants a massive `+5` points, heavily steering the evolution towards objective robustness.

### 4. Selection & Elitism
At the end of the tournament, the population is sorted.
- **Elitism**: The top 2 performing bots (the Champions) pass perfectly intact to the next generation. This mathematically guarantees that the population's maximum strength can never decrease.

### 5. Catastrophic Forgetting Prevention (The Archive)
A major issue in Genetic Algorithms is "Catastrophic Forgetting"—where Generation 10 forgets how to beat the strategies used in Generation 2.
- The absolute Champion of every generation is appended to a persistent `archive` array (The Museum).
- When breeding the next generation, one slot is randomly filled by a resurrected fossil from the Archive. The new generation *must* continuously prove it can defeat past ancestors.

### 6. Mutation
The remaining slots of the population are filled with "children" cloned from the Champions. These children undergo random mutation:
```python
# 20% chance to mutate a positional weight
if random.random() < 0.2:
    new_w[i] += random.randint(-10, 10)
```

---

## 📂 Repository Structure

```text
📦 Zola-AI-Engine
 ┣ 📂 docs/                # Project documentation and assignment rules
 ┣ 📂 logs/                # Training state and optimized genome JSONs
 ┃ ┣ 📜 training_log.txt
 ┃ ┣ 📜 training_state.json
 ┃ ┗ 📜 best_weights.json  # [REDACTED/IGNORED] The golden parameters
 ┣ 📂 scripts/
 ┃ ┗ 📜 train_zola.py      # Multi-core genetic optimizer
 ┣ 📂 src/
 ┃ ┣ 📜 ZolaGameS.py       # Core Game Logic & GUI
 ┃ ┣ 📜 playerAdvanced.py  # The AI Engine
 ┃ ┗ 📜 playerExampleAlpha.py # Baseline Opponent
 ┗ 📜 README.md
```

---

## 🚀 Usage

### 1. Running the Game
To launch the Graphical User Interface and play against the AI (or watch AI vs AI):
```bash
python src/ZolaGameS.py
```
You will be prompted to select the First Player and the timeout constraints. The `playerAdvanced` engine will automatically load its brain from `logs/best_weights.json`.

### 2. Running the Genetic Optimizer
To start the Genetic Algorithm and find new heuristic weights:
```bash
python scripts/train_zola.py
```
**Checkpointing & Graceful Exits**: The script features state-recovery. If you press `Ctrl+C` or hit the defined time limit, the exact state of the genetic pool is serialized to `training_state.json`. Running the command again will instantly resume the training from the exact generation you left off.

---

## 🔒 Competitive Integrity Note
*Some proprietary heuristics (e.g., precise bitwise masks) and the final trained weights (`best_weights.json`) have been obfuscated or added to `.gitignore` to maintain competitive integrity in academic tournaments. The architecture, however, remains fully functional for demonstration purposes.*
