import sys, os, time, random, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Mock dependencies for headless/CPU-only
import unittest.mock as mock
sys.modules['tkinter'] = mock.MagicMock()
sys.modules['tkinter.simpledialog'] = mock.MagicMock()
sys.modules['tkinter.messagebox'] = mock.MagicMock()
sys.modules['numpy'] = mock.MagicMock()
sys.modules['torch'] = mock.MagicMock()
sys.modules['playerGPU'] = mock.MagicMock()

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
            move = strategy(game, state, TIMEOUT)
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
    
    logs_dir = os.path.join(os.path.dirname(__file__), '../logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    out = os.path.join(logs_dir, 'training_positions.json')
    with open(out, 'w') as f:
        json.dump(all_data, f)
    print(f"Saved {len(all_data)} positions to {out}")

if __name__ == "__main__":
    main()
