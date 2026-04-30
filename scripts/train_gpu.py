import sys
import os
import random
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
except ImportError:
    print("PyTorch o NumPy non installati.")
    sys.exit(1)

from ZolaGameS import ZolaGame
import playerGPU as gpu_ai
import playerAdvanced as cpu_ai
from playerAdvanced import parse_state, generate_moves, make_move, sq_to_rc

def softmax(x, temp=1.0):
    x = np.array(x) / temp
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def play_game_against_cpu(model, device, gpu_is_red=True, temp=1.0):
    game = ZolaGame()
    state = game.initial
    states_history = []
    
    turns = 0
    while not game.is_terminal(state):
        if turns > 200:
            break
            
        red_bb, blue_bb, is_red_turn = parse_state(state)
        
        # Turno della GPU
        if (is_red_turn and gpu_is_red) or (not is_red_turn and not gpu_is_red):
            states_history.append((red_bb, blue_bb))
            
            legal_moves = generate_moves(red_bb, blue_bb, is_red_turn)
            if not legal_moves:
                move = None
            else:
                my_bbs = []
                opp_bbs = []
                for m in legal_moves:
                    n_red, n_blue, n_turn, _ = make_move(red_bb, blue_bb, is_red_turn, 0, m)
                    my_bbs.append(n_red)
                    opp_bbs.append(n_blue)
                    
                scores = gpu_ai.batched_evaluate(my_bbs, opp_bbs)
                
                # Softmax Exploration con Temperature
                if not is_red_turn:
                    scores = -scores # Inverti così massimizziamo sempre
                    
                probs = softmax(scores, temp)
                move_idx = np.random.choice(len(legal_moves), p=probs)
                move = legal_moves[move_idx]
        else:
            # Turno della CPU (Advanced Bot)
            # Diamo 0.5s alla CPU per velocizzare l'addestramento
            move = cpu_ai.playerStrategy(game, state, timeout=0.5)
            
        if move is None or move == "PASS":
            state = game.pass_turn(state)
        else:
            # Convert internal (sq_from, sq_to, is_cap) -> ((row,col), (row,col), is_cap)
            if isinstance(move[0], int):
                fr_sq, to_sq, is_cap = move
                move = (sq_to_rc(fr_sq), sq_to_rc(to_sq), is_cap)
            state = game.result(state, move)
            
        turns += 1
        
    winner = game.winner(state)
    # Calcolo Z dal punto di vista della GPU (1 = vittoria, -1 = sconfitta)
    if winner == "Red":
        z = 1.0 if gpu_is_red else -1.0
    elif winner == "Blue":
        z = -1.0 if gpu_is_red else 1.0
    else:
        z = 0.0
        
    return states_history, z

def train():
    if not gpu_ai.TORCH_AVAILABLE:
        print("PyTorch non disponibile. Impossibile procedere.")
        return
        
    model = gpu_ai.model
    device = gpu_ai.device
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f"Inizio Addestramento RL (Curriculum Learning vs CPU) su {device}")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=0.0, help="Minuti di training (0=infinito)")
    args = parser.parse_args()
    max_minutes = args.minutes
        
    start_time = time.time()
    batch_size = 128
    gamma = 0.99
    replay_buffer = []
    save_path = os.path.join(os.path.dirname(__file__), '../logs/gpu_weights.pth')
    
    gpu_is_red = True
    
    try:
        for game_idx in range(1000000):
            if max_minutes > 0 and (time.time() - start_time) / 60.0 > max_minutes:
                print("\n[Timer Scaduto] Tempo limite raggiunto!")
                break
                
            temp = max(0.1, 1.0 - (game_idx / 200.0))
            gpu_is_red = not gpu_is_red
            
            # Svuota la cache della CPU per non esplodere la RAM
            cpu_ai.TT.clear()
            
            states, z = play_game_against_cpu(model, device, gpu_is_red, temp)
            
            T = len(states)
            for t, (r_bb, b_bb) in enumerate(states):
                # La NN predice il vantaggio del Rosso.
                actual_target = z if gpu_is_red else -z
                target = actual_target * (gamma ** (T - 1 - t))
                replay_buffer.append((r_bb, b_bb, target))
                
            if len(replay_buffer) > 20000:
                replay_buffer = replay_buffer[-20000:]
                
            loss_val = 0.0
            if len(replay_buffer) >= batch_size:
                for _ in range(3):
                    batch = random.sample(replay_buffer, batch_size)
                    r_bbs = [b[0] for b in batch]
                    b_bbs = [b[1] for b in batch]
                    targets = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                    
                    inputs = gpu_ai.bitboards_to_tensor(r_bbs, b_bbs)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    loss_val = loss.item()
                    
            if (game_idx + 1) % 1 == 0:
                color = "Rosso" if gpu_is_red else "Blu"
                res = "Vinto" if z > 0 else ("Perso" if z < 0 else "Pari")
                print(f"Game {game_idx+1} | Giocato come: {color} | Risultato: {res} | Buffer: {len(replay_buffer)} | Temp: {temp:.2f} | Loss: {loss_val:.4f}")
                torch.save(model.state_dict(), save_path)
                
        print(f"\nAddestramento Completato! Pesi salvati in '{save_path}'")
        
    except KeyboardInterrupt:
        print("\n\n[Interrotto dall'utente] Salvo il modello...")
        torch.save(model.state_dict(), save_path)
        sys.exit(0)

if __name__ == "__main__":
    train()
