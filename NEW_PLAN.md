# Zola Champion: Implementation Plan

> **Note**: This implementation plan was designed by **Claude Opus 4.6**.
> The full implementation of the plan and the orchestration script were performed by **Gemini 3.1 Flash**.

---

## Code Review by Claude Opus 4.6

After reviewing every file produced by Gemini 3.1 Flash, I found and fixed the following issues:

### 🔴 CRITICAL: `timeout` vs `time_out` Parameter Mismatch

**Bug**: `player_TT_MO_KM.py` defines its function as `playerStrategy(game, state, time_out)` — using `time_out` (with underscore) instead of the `timeout` used by every other player. Gemini's scripts all called it with `timeout=TIMEOUT` as a keyword argument, which would **crash with a TypeError** whenever the opponent needed to make a move.

**Affected files**: `collect_positions.py`, `validate_champion.py`, `train_champion_ga.py`

**Fix**: Changed all strategy calls to use **positional arguments** (`strategy(game, state, TIMEOUT)`) instead of keyword arguments. This works with both `timeout` and `time_out` parameter names.

### 🟡 MEDIUM: Texel Tuner Ignoring Player Perspective

**Bug**: `texel_tune.py`'s `fast_evaluate()` always used `weights[33]` for material — the Red player's weight. But the actual engine (`playerChampion.evaluate()`) selects `weights[0:34]` for Red or `weights[34:68]` for Blue based on `is_red_turn`. This means the Texel tuner was only optimizing the Red-player weights and ignoring the Blue-player half entirely — a subtle but devastating training bug.

**Fix**: Added `is_red_turn` parameter to `fast_evaluate()`, made it select the correct weight slice, and fixed `compute_error()` to pass the turn flag and normalize perspective for the sigmoid.

### 🟢 MINOR: Orchestrator Working Directory

**Bug**: `run_full_training.py` used `subprocess.Popen(command, shell=True)` without specifying a working directory. If run from any directory other than the project root, the relative paths (`scripts/collect_positions.py`) would fail.

**Fix**: Added `PROJECT_ROOT` detection and passed `cwd=PROJECT_ROOT` to `Popen`. Also added validation as a 4th automatic step.

### ✅ Verified Correct

- `playerChampion.py` — PVS, Quiescence, LMR, History, Null Move, Aspiration Windows all implement correctly
- Bitboard engine, Zobrist hashing, TT logic preserved correctly from playerElite
- Move ordering hierarchy correct (TT > Captures > Killers > History)
- Weight loading from `best_weights_champion.json` works
- 4-game integration test passed (0W-4L with zero weights, as expected — training is required)

---



---

## PART A: SEARCH ENGINE IMPROVEMENTS (playerChampion.py)

Each step below is a self-contained task. Apply them in order to `src/playerChampion.py`.

---

### Step 1: Fork playerElite.py → playerChampion.py

**What**: Copy `src/playerElite.py` to `src/playerChampion.py`. Change the weights file path.

**Exact changes**:
1. `cp src/playerElite.py src/playerChampion.py`
2. In `load_best_weights()`, change the path from `'../logs/best_weights_elite.json'` to `'../logs/best_weights_champion.json'`
3. In the print/log at the end of `playerStrategy`, add a tag: `# print(f"[Champion] depth={depth}")`
4. Verify it runs: import it and call `playerStrategy(game, state)` — should work identically to Elite.

**Acceptance**: `playerChampion.playerStrategy(game, state, timeout=3)` returns a valid move.

---

### Step 2: Add Principal Variation Search (PVS)

**What**: Replace the plain negamax loop with PVS. The first move (expected best) gets a full-window search. All subsequent moves get a zero-window scout search. If the scout finds something better, re-search with full window.

**Why**: PVS prunes ~15-30% more nodes than plain alpha-beta when move ordering is good (and ours is — TT+Killers+Captures).

**Where**: Modify the `negamax()` function in `playerChampion.py`, specifically the move loop (lines ~216-229).

**Current code** (simplified):
```python
for move in moves:
    nr, nb, nt, nh = make_move(...)
    val, _ = negamax(nr, nb, nt, depth-1, -beta, -alpha, nh, ...)
    val = -val
    if val > best_v: best_v, best_m = val, move
    alpha = max(alpha, val)
    if alpha >= beta:
        # killer update
        break
```

**Replace with**:
```python
for i, move in enumerate(moves):
    nr, nb, nt, nh = make_move(...)
    
    if i == 0:
        # Full window search for PV move
        val, _ = negamax(nr, nb, nt, depth-1, -beta, -alpha, nh, start_time, time_limit, ply+1)
        val = -val
    else:
        # Zero-window scout search
        val, _ = negamax(nr, nb, nt, depth-1, -alpha-1, -alpha, nh, start_time, time_limit, ply+1)
        val = -val
        if alpha < val < beta:
            # Scout found something better — re-search with full window
            val, _ = negamax(nr, nb, nt, depth-1, -beta, -val, nh, start_time, time_limit, ply+1)
            val = -val
    
    if val > best_v: best_v, best_m = val, move
    alpha = max(alpha, val)
    if alpha >= beta:
        if not move[2] and ply < 100:
            if KILLERS[ply][0] != move:
                KILLERS[ply][1] = KILLERS[ply][0]
                KILLERS[ply][0] = move
        break
```

**Acceptance**: Run a game vs TT_MO_KM at 3s timeout. Champion should reach equal or greater depth than before (PVS should never reduce depth, only improve it).

### Step 3: Add Quiescence Search

**What**: When the main search reaches `depth == 0`, instead of returning the static eval immediately, continue searching **capture moves only** until the position is "quiet" (no more captures available). This prevents the horizon effect where the engine evaluates right before an obvious capture.

**Why**: This is the **single biggest improvement** possible. Neither TT_MO_KM nor any existing player has it. It prevents the bot from walking into obvious traps because it stopped searching 1 ply too early.

**Where**: Add a new function `quiescence()` and modify the `depth <= 0` check in `negamax()`.

**Add this new function** (place it right before `negamax()`):
```python
MAX_QS_DEPTH = 6  # Limit quiescence to 6 extra plies

def quiescence(red_bb, blue_bb, is_red_turn, alpha, beta, hash_val, start_time, time_limit, qs_depth):
    """Search captures only until position is quiet."""
    if time.time() - start_time > time_limit:
        raise TimeoutError()
    
    # Terminal checks
    if red_bb == 0: return -10000000
    if blue_bb == 0: return 10000000
    
    # Stand-pat: the static eval is a lower bound (we can always choose not to capture)
    stand_pat = evaluate(red_bb, blue_bb, is_red_turn)
    
    if qs_depth <= 0:
        return stand_pat
    
    if stand_pat >= beta:
        return beta  # Beta cutoff — position is already too good
    if stand_pat > alpha:
        alpha = stand_pat  # Raise alpha to stand-pat
    
    # Generate ONLY capture moves
    moves = generate_moves(red_bb, blue_bb, is_red_turn)
    captures = [m for m in moves if m[2]]  # m[2] is is_capture flag
    
    if not captures:
        return stand_pat  # No captures — position is quiet
    
    for move in captures:
        nr, nb, nt, nh = make_move(red_bb, blue_bb, is_red_turn, hash_val, move)
        val = -quiescence(nr, nb, nt, -beta, -alpha, nh, start_time, time_limit, qs_depth - 1)
        
        if val >= beta:
            return beta
        if val > alpha:
            alpha = val
    
    return alpha
```

**Modify `negamax()`** — change the `depth <= 0` block:

**Current**:
```python
if depth <= 0: return evaluate(red_bb, blue_bb, is_red_turn), None
```

**Replace with**:
```python
if depth <= 0:
    val = quiescence(red_bb, blue_bb, is_red_turn, alpha, beta, hash_val, start_time, time_limit, MAX_QS_DEPTH)
    return val, None
```

**Acceptance**: Play 5 games vs TT_MO_KM. The champion should no longer walk into obvious 1-move captures that TT_MO_KM could see. Search depth may decrease by 1 ply (since leaves are now deeper), but move quality should be noticeably better.

### Step 4: Add Late Move Reduction (LMR)

**What**: Moves that appear late in the move ordering (not TT move, not capture, not killer) are unlikely to be good. Search them at reduced depth (`depth - 2` instead of `depth - 1`). If they surprise us by scoring above alpha, re-search at full depth.

**Why**: Allows the engine to search 1-2 plies deeper within the same time budget. Huge impact in Zola where the branching factor is high (~20-40 moves per position).

**Where**: Modify the PVS move loop inside `negamax()` (the code from Step 2).

**Conditions for reduction** (ALL must be true):
- `i >= 3` (not one of the first 3 moves — those are already well-ordered)
- `depth >= 3` (don't reduce at shallow depths)
- `not move[2]` (not a capture)
- `move != KILLERS[ply][0] and move != KILLERS[ply][1]` (not a killer)

**Modified PVS loop** (replaces the Step 2 code):
```python
for i, move in enumerate(moves):
    nr, nb, nt, nh = make_move(red_bb, blue_bb, is_red_turn, hash_val, move)
    
    if i == 0:
        val, _ = negamax(nr, nb, nt, depth-1, -beta, -alpha, nh, start_time, time_limit, ply+1)
        val = -val
    else:
        # LMR: reduce depth for late quiet moves
        reduction = 0
        if (i >= 3 and depth >= 3 and not move[2] 
                and move != KILLERS[ply][0] and move != KILLERS[ply][1]):
            reduction = 1  # Search at depth-2 instead of depth-1
        
        # Zero-window scout with possible reduction
        val, _ = negamax(nr, nb, nt, depth-1-reduction, -alpha-1, -alpha, nh, start_time, time_limit, ply+1)
        val = -val
        
        # If reduced search found something interesting, re-search at full depth
        if reduction > 0 and val > alpha:
            val, _ = negamax(nr, nb, nt, depth-1, -alpha-1, -alpha, nh, start_time, time_limit, ply+1)
            val = -val
        
        # If scout found something better, full window re-search
        if alpha < val < beta:
            val, _ = negamax(nr, nb, nt, depth-1, -beta, -val, nh, start_time, time_limit, ply+1)
            val = -val
    
    if val > best_v: best_v, best_m = val, move
    alpha = max(alpha, val)
    if alpha >= beta:
        if not move[2] and ply < 100:
            if KILLERS[ply][0] != move:
                KILLERS[ply][1] = KILLERS[ply][0]
                KILLERS[ply][0] = move
        break
```

**Acceptance**: Compare max depth reached in 3s vs the Step 2 version. Should consistently reach 1+ ply deeper.

### Step 5: Add History Heuristic

**What**: Track which `(from_sq, to_sq)` pairs cause beta cutoffs across the entire search. Use this as a secondary ordering signal after TT/Captures/Killers.

**Where**: Add a global `HISTORY` table and modify move ordering in `negamax()`.

**Add at module level** (near KILLERS):
```python
HISTORY = [[0] * 64 for _ in range(64)]  # HISTORY[from_sq][to_sq]
```

**Modify the killer update section** (when beta cutoff happens on a quiet move):
```python
if alpha >= beta:
    if not move[2] and ply < 100:
        # Killer update (existing)
        if KILLERS[ply][0] != move:
            KILLERS[ply][1] = KILLERS[ply][0]
            KILLERS[ply][0] = move
        # History update (NEW)
        HISTORY[move[0]][move[1]] += depth * depth  # Deeper cutoffs worth more
    break
```

**Modify move ordering** — update `m_score()`:
```python
k1, k2 = KILLERS[ply][0], KILLERS[ply][1]
def m_score(m):
    if m == tt_move: return 100000
    if m[2]: return 10000          # Captures
    if m == k1: return 5000        # Killer 1
    if m == k2: return 4000        # Killer 2
    return HISTORY[m[0]][m[1]]     # History score (NEW — replaces 0)
```

**Reset HISTORY** in `playerStrategy()` alongside KILLERS:
```python
for row in HISTORY: 
    for j in range(64): row[j] = 0
```

**Acceptance**: No functional change expected, but depth should increase slightly due to better move ordering.

---

### Step 6: Add Null Move Pruning

**What**: Before searching normally, try "passing" (giving the opponent a free move). If the opponent still can't beat beta even with a free move, this position is so good we can prune it. Use a reduced depth search (`depth - 3`).

**Why**: In winning positions, this cuts the tree dramatically. The opponent can't do enough damage even with a free turn.

**Where**: Add to `negamax()`, right after the TT probe and before move generation.

**Add this block** (after `if depth <= 0: ...` and before `moves = generate_moves(...)`):
```python
# Null Move Pruning (skip in endgame to avoid zugzwang)
total_pieces = red_bb.bit_count() + blue_bb.bit_count()
if depth >= 3 and total_pieces > 8 and not is_pv:
    # Give opponent a free move and search at reduced depth
    null_hash = hash_val ^ ZOBRIST_TURN
    null_val, _ = negamax(red_bb, blue_bb, not is_red_turn, depth - 3, -beta, -beta + 1, 
                          null_hash, start_time, time_limit, ply + 1)
    null_val = -null_val
    if null_val >= beta:
        return beta, None  # Prune — position is too good
```

**Important**: Add `is_pv` parameter to `negamax()` signature:
```python
def negamax(red_bb, blue_bb, is_red_turn, depth, alpha, beta, hash_val, 
            start_time, time_limit, ply, is_pv=True):
```

In the PVS loop, pass `is_pv=False` for scout searches and `is_pv=True` for full-window re-searches. The `i == 0` full-window call should pass `is_pv=True`.

**Acceptance**: In positions where Champion is clearly winning (material advantage ≥ 3), search should be noticeably faster (higher depth reached).

---

### Step 7: Add Aspiration Windows

**What**: In iterative deepening, use the previous depth's score ± a window instead of `(-∞, +∞)`. If the search fails outside the window, widen and re-search.

**Where**: Modify the iterative deepening loop in `playerStrategy()`.

**Current**:
```python
while True:
    try:
        val, move = negamax(..., depth, -math.inf, math.inf, ...)
        if move: best_move = move
        ...
        depth += 1
    except TimeoutError: break
```

**Replace with**:
```python
prev_score = 0
WINDOW = 50

while True:
    try:
        # Aspiration window based on previous depth's score
        if depth <= 1:
            a, b = -math.inf, math.inf
        else:
            a, b = prev_score - WINDOW, prev_score + WINDOW
        
        val, move = negamax(..., depth, a, b, ..., is_pv=True)
        
        # If score fell outside window, re-search with full bounds
        if val <= a or val >= b:
            val, move = negamax(..., depth, -math.inf, math.inf, ..., is_pv=True)
        
        if move: best_move = move
        prev_score = val
        if val > 9000000 or val < -9000000: break
        depth += 1
    except TimeoutError: break
```

**Acceptance**: No strength regression. In stable positions, should save ~10-20% time per iteration.

---

## PART B: TRAINING PIPELINE (train_champion.py)

### Step 8: Position Collection Script

**What**: Create `scripts/collect_positions.py` — plays fast games between existing bots and saves every board position + game outcome to a JSON file. This data is used by Texel Tuning (Step 9).

**Where**: New file `scripts/collect_positions.py`

**How it works**:
1. Play 200 fast games (0.3s/move timeout) between pairs: Champion vs TT_MO_KM, Champion vs Advanced, self-play
2. For each game, record every position as `(red_bb, blue_bb, is_red_turn)` and the final result (`1.0` = Red wins, `0.0` = Blue wins, `0.5` = draw/timeout)
3. Save to `logs/training_positions.json`

**Output format**:
```json
[
  {"red_bb": 123456, "blue_bb": 789012, "is_red_turn": true, "result": 1.0},
  ...
]
```

**Implementation sketch**:
```python
import sys, os, time, random, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Mock tkinter for headless
import unittest.mock as mock
sys.modules['tkinter'] = mock.MagicMock()
sys.modules['tkinter.simpledialog'] = mock.MagicMock()
sys.modules['tkinter.messagebox'] = mock.MagicMock()

from ZolaGameS import ZolaGame
import playerChampion as champ
import player_TT_MO_KM as opponent
import playerAdvanced as advanced
from playerChampion import parse_state, generate_moves, BIT_MASKS

TIMEOUT = 0.3
MAX_TURNS = 150

def play_and_collect(p1_strategy, p2_strategy, game):
    """Play one game, return list of (red_bb, blue_bb, is_red_turn) and result."""
    state = game.initial
    positions = []
    turns = 0
    while not game.is_terminal(state) and turns < MAX_TURNS:
        rb, bb, is_red = parse_state(state)
        positions.append((rb, bb, is_red))
        
        legal = game.actions(state)
        if not legal:
            state = game.pass_turn(state)
            turns += 1
            continue
        
        strategy = p1_strategy if state.to_move == "Red" else p2_strategy
        try:
            move = strategy(game, state, timeout=TIMEOUT)
        except:
            move = None
        if move not in legal:
            move = random.choice(legal)
        state = game.result(state, move)
        turns += 1
    
    winner = game.winner(state)
    result = 1.0 if winner == "Red" else (0.0 if winner == "Blue" else 0.5)
    return positions, result

def main():
    game = ZolaGame()
    all_data = []
    
    matchups = [
        (champ.playerStrategy, opponent.playerStrategy, "Champ vs TT_MO_KM"),
        (opponent.playerStrategy, champ.playerStrategy, "TT_MO_KM vs Champ"),
        (champ.playerStrategy, advanced.playerStrategy, "Champ vs Advanced"),
        (champ.playerStrategy, champ.playerStrategy, "Self-play"),
    ]
    
    for p1, p2, name in matchups:
        for i in range(50):
            positions, result = play_and_collect(p1, p2, game)
            for rb, bb, is_red in positions:
                all_data.append({"red_bb": rb, "blue_bb": bb, "is_red_turn": is_red, "result": result})
            print(f"{name} game {i+1}/50 done, {len(positions)} positions, result={result}")
    
    out = os.path.join(os.path.dirname(__file__), '../logs/training_positions.json')
    with open(out, 'w') as f:
        json.dump(all_data, f)
    print(f"Saved {len(all_data)} positions to {out}")

if __name__ == "__main__":
    main()
```

**Expected runtime**: ~30-60 minutes for 200 games at 0.3s/move. Produces ~10,000-15,000 positions.

**Acceptance**: `logs/training_positions.json` exists with >5,000 entries.

---

### Step 9: Texel Tuning Optimizer

**What**: Create `scripts/texel_tune.py` — reads the positions from Step 8 and finds the 68 weights that best predict game outcomes using gradient-free optimization. **No games are played during optimization** — this is pure math on pre-collected data.

**Where**: New file `scripts/texel_tune.py`

**The Texel formula**: For each position, compute:
```
error = (result - sigmoid(K * eval(position, weights)))²
```
Where `K` is a scaling constant (~0.004), `sigmoid(x) = 1 / (1 + exp(-x))`, and `result` is the actual game outcome (1.0/0.0/0.5). Minimize the mean error across all positions.

**Implementation**:
```python
import json, os, sys, math, random, time
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from playerChampion import LEVELS, BIT_MASKS

K = 0.004  # Scaling constant — tune if needed

def sigmoid(x):
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def fast_evaluate(red_bb, blue_bb, weights):
    """Evaluate from Red's perspective using the 68-param scheme."""
    rc = red_bb.bit_count()
    bc = blue_bb.bit_count()
    if bc == 0: return 10000000
    if rc == 0: return -10000000
    
    total = rc + bc
    phase_offset = 0 if total > 48 else (10 if total > 24 else 20)
    
    # Material
    score = (rc - bc) * weights[33]
    
    # Positional
    r_bb = red_bb
    while r_bb:
        lsb = r_bb & -r_bb
        sq = lsb.bit_length() - 1
        score += weights[phase_offset + LEVELS[sq]]
        r_bb ^= lsb
    
    b_bb = blue_bb
    while b_bb:
        lsb = b_bb & -b_bb
        sq = lsb.bit_length() - 1
        score -= weights[phase_offset + LEVELS[sq]]
        b_bb ^= lsb
    
    # Clustering
    red_cl = (red_bb & (red_bb << 1)).bit_count() + (red_bb & (red_bb >> 8)).bit_count()
    blue_cl = (blue_bb & (blue_bb << 1)).bit_count() + (blue_bb & (blue_bb >> 8)).bit_count()
    score += (red_cl - blue_cl) * weights[31]
    
    return score

def compute_error(positions, weights):
    """Mean squared error across all positions."""
    total_error = 0.0
    for pos in positions:
        ev = fast_evaluate(pos["red_bb"], pos["blue_bb"], weights)
        predicted = sigmoid(K * ev)
        total_error += (pos["result"] - predicted) ** 2
    return total_error / len(positions)

def optimize(positions, initial_weights, iterations=5000):
    """Hill-climbing optimizer: mutate one weight at a time, keep if error decreases."""
    best_w = list(initial_weights)
    best_error = compute_error(positions, best_w)
    
    # Which indices to tune (skip index 0 and unused phase boundaries)
    tunable = list(range(1, 10)) + list(range(11, 20)) + list(range(21, 30)) + [31, 33]
    # Mirror for second player (offset 34)
    tunable += [i + 34 for i in tunable]
    
    for iteration in range(iterations):
        idx = random.choice(tunable)
        
        # Try +delta and -delta
        for delta in [1, -1, 5, -5, 20, -20, 50, -50, 100, -100]:
            trial_w = list(best_w)
            trial_w[idx] += delta
            trial_error = compute_error(positions, trial_w)
            if trial_error < best_error:
                best_w = trial_w
                best_error = trial_error
                break
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}/{iterations} | Error: {best_error:.6f}")
    
    return best_w, best_error

def main():
    pos_file = os.path.join(os.path.dirname(__file__), '../logs/training_positions.json')
    with open(pos_file, 'r') as f:
        positions = json.load(f)
    
    print(f"Loaded {len(positions)} positions")
    
    # Start from current best or default
    initial = [0] * 68
    initial[33] = 1000  # Material weight for Red
    initial[67] = 1000  # Material weight for Blue
    
    best_w, best_error = optimize(positions, initial, iterations=10000)
    
    out = os.path.join(os.path.dirname(__file__), '../logs/best_weights_champion.json')
    with open(out, 'w') as f:
        json.dump({"weights": best_w, "error": best_error}, f, indent=2)
    print(f"Best error: {best_error:.6f}")
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
```

**Expected runtime**: 10,000 iterations on 10,000 positions ≈ **5-15 minutes** on CPU.

**Acceptance**: `logs/best_weights_champion.json` exists with 68 weights and error < initial error.

---

### Step 10: Ultra-Lean GA Trainer (Refinement)

**What**: Create `scripts/train_champion_ga.py` — a genetic algorithm that plays actual games to refine weights after Texel tuning. Uses the Texel-tuned weights as the starting point.

**Where**: New file `scripts/train_champion_ga.py`

**Key design for speed**:
- Population: **12** individuals
- Seeded from: Texel-tuned weights (Step 9) + 11 mutants
- Each individual plays **only 4 games** per generation: Red vs TT_MO_KM, Blue vs TT_MO_KM, Red vs Advanced, Blue vs Advanced
- **No round-robin** — saves N² games
- Timeout per move: **0.1s** (depth 2-3 is enough to judge weight quality)
- Max 150 turns per game
- Scoring: Win vs TT_MO_KM = **10 pts**, Win vs Advanced = **5 pts**, Draw = 1 pt
- Elitism: top 3 survive, 9 children via mutation + crossover
- Mutation: 25% chance per weight, range ±30 for positional, ±200 for material
- **Parallel**: Use `ProcessPoolExecutor` with `cpu_count() - 1` workers

**Expected speed**: 48 games/gen × ~15s/game ÷ 8 cores ≈ **~90 seconds per generation**. In 12 hours: **~480 generations**.

**Implementation**: Follow the same pattern as `train_elite.py` but with the above parameters. The key differences are:
1. Import `playerChampion` instead of `playerElite`
2. Import `player_TT_MO_KM` as the main opponent
3. No round-robin — only play vs fixed opponents
4. Load initial weights from `best_weights_champion.json` (Texel output)
5. Save to `best_weights_champion.json` (overwrite with better weights)

**Acceptance**: After 50+ generations, the champion should beat TT_MO_KM in >50% of training games.

---

### Step 11: Validation Tournament Script

**What**: Create `scripts/validate_champion.py` — plays 20 games Champion vs TT_MO_KM at **3.0s timeout** (tournament conditions) and reports win/loss/draw.

**Where**: New file `scripts/validate_champion.py`

**Key details**:
- Alternate who plays Red/Blue each game
- 3.0s timeout (real tournament conditions)
- Clear TT between games
- Print live progress dashboard
- Final summary: win rate, avg depth reached, avg time per move

**Implementation**: Follow the pattern of `scripts/run_tournament.py` but simplified:
```python
# ... (standard imports and tkinter mock)

from ZolaGameS import ZolaGame
import playerChampion as champ
import player_TT_MO_KM as opponent

NUM_GAMES = 20
TIMEOUT = 3.0

def run():
    wins, losses, draws = 0, 0, 0
    
    for game_num in range(NUM_GAMES):
        champ_is_red = (game_num % 2 == 0)
        game = ZolaGame()
        state = game.initial
        champ.TT.clear()
        opponent.TT.clear()
        turns = 0
        
        while not game.is_terminal(state) and turns < 200:
            legal = game.actions(state)
            if not legal:
                state = game.pass_turn(state)
                turns += 1
                continue
            
            is_champ_turn = (state.to_move == "Red") == champ_is_red
            strategy = champ.playerStrategy if is_champ_turn else opponent.playerStrategy
            
            try:
                move = strategy(game, state, timeout=TIMEOUT)
            except:
                move = None
            if move not in legal:
                move = random.choice(legal)
            state = game.result(state, move)
            turns += 1
        
        winner = game.winner(state)
        champ_color = "Red" if champ_is_red else "Blue"
        if winner == champ_color:
            wins += 1; result = "WIN"
        elif winner is None:
            draws += 1; result = "DRAW"
        else:
            losses += 1; result = "LOSS"
        
        print(f"Game {game_num+1}/{NUM_GAMES}: Champion={champ_color} → {result} | "
              f"Record: {wins}W-{losses}L-{draws}D")
    
    wr = wins / NUM_GAMES * 100
    print(f"\nFINAL: {wins}W-{losses}L-{draws}D ({wr:.0f}% win rate)")
    print("TARGET: >60% win rate to consider successful")

if __name__ == "__main__":
    run()
```

**Acceptance**: Script runs to completion. Target: **>60% win rate** vs TT_MO_KM.

---

## PART C: EXECUTION TIMELINE

```
Hour 0-3:    Implement Steps 1-7 (playerChampion.py with all search improvements)
Hour 3-4:    Implement Step 8 (collect_positions.py) + run it (~30 min)
Hour 4-5:    Implement Step 9 (texel_tune.py) + run it (~15 min)
Hour 5-6:    Implement Step 10 (train_champion_ga.py) + start it
Hour 6:      Implement Step 11 (validate_champion.py)
Hour 6-7:    Run quick validation (5 games). Fix bugs if any.
Hour 7-19:   GA training runs overnight unattended (12 hours ≈ 480 generations)
Hour 19-20:  Morning: run full 20-game validation at 3s timeout
Hour 20:     Final adjustments if needed. Submit.
```

**Critical path**: Steps 1-7 must be done first (the engine). Training can only start once the engine works correctly.

---

## QUICK REFERENCE: File Map

| File | Purpose | Status |
|---|---|---|
| `src/playerChampion.py` | Tournament submission player | NEW (Steps 1-7) |
| `scripts/collect_positions.py` | Generate training data | NEW (Step 8) |
| `scripts/texel_tune.py` | Fast weight optimization | NEW (Step 9) |
| `scripts/train_champion_ga.py` | Genetic refinement trainer | NEW (Step 10) |
| `scripts/validate_champion.py` | Tournament validation | NEW (Step 11) |
| `logs/training_positions.json` | Position dataset | Generated by Step 8 |
| `logs/best_weights_champion.json` | Trained weights | Generated by Steps 9-10 |
| `src/playerAdvanced.py` | ⛔ DO NOT MODIFY | Existing |
| `src/player_TT_MO_KM.py` | The opponent to beat | Existing (read-only) |
