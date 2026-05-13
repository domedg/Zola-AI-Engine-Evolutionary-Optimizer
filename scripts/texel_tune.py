import json, os, sys, math, random, time
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from playerChampion import LEVELS, BIT_MASKS

K = 0.004  # Scaling constant — tune if needed

def sigmoid(x):
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def fast_evaluate(red_bb, blue_bb, is_red_turn, weights):
    """Evaluate from current player's perspective using the 68-param scheme.
    Mirrors playerChampion.evaluate() exactly."""
    rc = red_bb.bit_count()
    bc = blue_bb.bit_count()
    if bc == 0: return 10000000
    if rc == 0: return -10000000
    
    # Select correct weight slice based on who we are
    w = weights[0:34] if is_red_turn else weights[34:68]
    
    total = rc + bc
    phase_offset = 0 if total > 48 else (10 if total > 24 else 20)
    
    # Material
    score = (rc - bc) * w[33]
    
    # Positional
    r_bb = red_bb
    while r_bb:
        lsb = r_bb & -r_bb
        sq = lsb.bit_length() - 1
        score += w[phase_offset + LEVELS[sq]]
        r_bb ^= lsb
    
    b_bb = blue_bb
    while b_bb:
        lsb = b_bb & -b_bb
        sq = lsb.bit_length() - 1
        score -= w[phase_offset + LEVELS[sq]]
        b_bb ^= lsb
    
    # Clustering
    red_cl = (red_bb & (red_bb << 1)).bit_count() + (red_bb & (red_bb >> 8)).bit_count()
    blue_cl = (blue_bb & (blue_bb << 1)).bit_count() + (blue_bb & (blue_bb >> 8)).bit_count()
    score += (red_cl - blue_cl) * w[31]
    
    # Return from current player's perspective
    return score if is_red_turn else -score

def compute_error(positions, weights):
    """Mean squared error across all positions."""
    total_error = 0.0
    for pos in positions:
        is_red = pos["is_red_turn"]
        ev = fast_evaluate(pos["red_bb"], pos["blue_bb"], is_red, weights)
        # For Texel: sigmoid should predict result from RED's perspective
        # If it's Blue's turn, negate eval to get Red's perspective
        ev_red = ev if is_red else -ev
        predicted = sigmoid(K * ev_red)
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
    if not os.path.exists(pos_file):
        print(f"Error: {pos_file} not found. Run collect_positions.py first.")
        return
        
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
