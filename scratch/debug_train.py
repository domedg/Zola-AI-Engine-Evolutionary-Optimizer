import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import torch
import numpy as np
from ZolaGameS import ZolaGame
import playerGPU as gpu_ai
import playerAdvanced as cpu_ai
from playerAdvanced import parse_state, generate_moves, make_move, sq_to_rc

print("Checking modules...")
print(f"gpu_ai: {gpu_ai}")
print(f"cpu_ai: {cpu_ai}")

game = ZolaGame()
state = game.initial
print("Game initialized.")

turns = 0
while not game.is_terminal(state) and turns < 10:
    print(f"Turn {turns}")
    red_bb, blue_bb, is_red_turn = parse_state(state)
    
    if is_red_turn:
        print("GPU Turn (simulated)")
        legal_moves = generate_moves(red_bb, blue_bb, is_red_turn)
        move = legal_moves[0] if legal_moves else None
    else:
        print("CPU Turn (Advanced)")
        move = cpu_ai.playerStrategy(game, state, timeout=0.1)
    
    if move:
        print(f"Move: {move}")
        if isinstance(move[0], int):
            fr_sq, to_sq, is_cap = move
            move = (sq_to_rc(fr_sq), sq_to_rc(to_sq), is_cap)
        state = game.result(state, move)
    else:
        state = game.pass_turn(state)
    turns += 1

print("Test finished successfully.")
