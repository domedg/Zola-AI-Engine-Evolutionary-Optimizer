import sys
import os
import time
from collections import defaultdict
import unittest.mock as mock

# Mock tkinter per runnare la logica del gioco in modo headless
sys.modules['tkinter'] = mock.MagicMock()
sys.modules['tkinter.simpledialog'] = mock.MagicMock()
sys.modules['tkinter.messagebox'] = mock.MagicMock()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ZolaGameS import ZolaGame
import playerGPU as gpu_ai
import playerAdvanced as cpu_ai

def print_dashboard(match_idx, total_matches, stats):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*50)
    print(" ZOLA AI - TOURNAMENT ANALYTICS DASHBOARD")
    print("="*50)
    
    progress = int((match_idx / total_matches) * 30)
    bar = "█" * progress + "-" * (30 - progress)
    print(f" Progress: [{bar}] {match_idx}/{total_matches}")
    print("-"*50)
    
    gpu_wins = stats['gpu_wins']
    cpu_wins = stats['cpu_wins']
    draws = stats['draws']
    
    gpu_wr = (gpu_wins / match_idx * 100) if match_idx > 0 else 0
    cpu_wr = (cpu_wins / match_idx * 100) if match_idx > 0 else 0
    
    print(f" [GPU ResNet Bot]   Wins: {gpu_wins:3d}  |  Win Rate: {gpu_wr:5.1f}%")
    print(f" [CPU Advanced Bot] Wins: {cpu_wins:3d}  |  Win Rate: {cpu_wr:5.1f}%")
    print(f" Draws: {draws}")
    print("-"*50)
    
    avg_gpu_time = sum(stats['gpu_times']) / len(stats['gpu_times']) if stats['gpu_times'] else 0
    avg_cpu_time = sum(stats['cpu_times']) / len(stats['cpu_times']) if stats['cpu_times'] else 0
    
    print(f" Avg Time/Move (GPU): {avg_gpu_time:.3f}s")
    print(f" Avg Time/Move (CPU): {avg_cpu_time:.3f}s")
    print("="*50)

def run_tournament():
    try:
        num_matches = int(input("Quante partite vuoi simulare nel torneo? [default 10]: ") or "10")
        timeout = float(input("Timeout per mossa (secondi)? [default 3.0]: ") or "3.0")
    except ValueError:
        num_matches = 10
        timeout = 3.0
        
    stats = {
        'gpu_wins': 0,
        'cpu_wins': 0,
        'draws': 0,
        'gpu_times': [],
        'cpu_times': []
    }
    
    gpu_is_red = True
    
    for match_idx in range(1, num_matches + 1):
        game = ZolaGame()
        state = game.initial
        gpu_is_red = not gpu_is_red
        
        cpu_ai.TT.clear() # Prevent memory leak between matches
        
        turns = 0
        while not game.is_terminal(state):
            if turns > 200:
                break
                
            if (state.to_move == "Red" and gpu_is_red) or (state.to_move == "Blue" and not gpu_is_red):
                start = time.time()
                move = gpu_ai.playerStrategy(game, state, timeout=timeout)
                dur = time.time() - start
                stats['gpu_times'].append(dur)
            else:
                start = time.time()
                move = cpu_ai.playerStrategy(game, state, timeout=timeout)
                dur = time.time() - start
                stats['cpu_times'].append(dur)
                
            if move is None or move == "PASS":
                state = game.pass_turn(state)
            else:
                state = game.result(state, move)
                
            turns += 1
            
            # Update Dashboard every 10 turns to show live progress
            if turns % 10 == 0:
                print_dashboard(match_idx - 1, num_matches, stats)
                
        winner = game.winner(state)
        if winner == "Red":
            if gpu_is_red: stats['gpu_wins'] += 1
            else: stats['cpu_wins'] += 1
        elif winner == "Blue":
            if not gpu_is_red: stats['gpu_wins'] += 1
            else: stats['cpu_wins'] += 1
        else:
            stats['draws'] += 1
            
        print_dashboard(match_idx, num_matches, stats)
        
if __name__ == "__main__":
    run_tournament()
