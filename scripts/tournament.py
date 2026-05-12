import sys
import os
import time
import random
import copy

# Aggiunge src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ZolaGameS import ZolaGame
import playerAdvanced
import player_TT_MO_KM
import playerPro130
import playerExampleAlpha
import playerGPU

def run_match(player1_mod, player2_mod, name1, name2):
    game = ZolaGame()
    state = game.initial
    turns = 0
    # Timeout per mossa aumentato per vedere la vera forza
    MOVE_TIMEOUT = 3.0 
    
    while not game.is_terminal(state):
        if turns > 200: return "Pareggio (Loop)"
        current_player = state.to_move
        strategy = player1_mod.playerStrategy if current_player == "Red" else player2_mod.playerStrategy
        
        state_copy = copy.deepcopy(state)
        try:
            move = strategy(game, state_copy, timeout=MOVE_TIMEOUT)
        except Exception:
            move = None
            
        legal = game.actions(state)
        if not legal:
            try:
                state = game.pass_turn(state)
            except:
                state.to_move = "Blue" if state.to_move == "Red" else "Red"
        else:
            if move not in legal:
                move = random.choice(legal)
            state = game.result(state, move)
        turns += 1
    
    winner = game.winner(state)
    return name1 if winner == "Red" else name2

def main():
    players = [
        (playerAdvanced, "Advanced (68)"),
        (player_TT_MO_KM, "Collega (TT-MO-KM)"),
        (playerPro130, "Pro130 (Elite)"),
        (playerExampleAlpha, "Alpha (Baseline)"),
        (playerGPU, "Zola-GPU (Neural)")
    ]
    
    print("\n" + "="*50)
    print("🏟️  GRAN TORNEO ZOLA AI: SFIDA DEI 3 SECONDI")
    print("="*50)
    
    results = {name: 0 for _, name in players}
    
    for i, (mod1, name1) in enumerate(players):
        for j, (mod2, name2) in enumerate(players):
            if i == j: continue
            
            print(f"⚔️  {name1:20} vs {name2:20} ...", end=" ", flush=True)
            start_t = time.time()
            try:
                winner = run_match(mod1, mod2, name1, name2)
                dur = time.time() - start_t
                print(f"VINCITORE: {winner:20} ({dur:.1f}s)")
                if winner in results: results[winner] += 1
            except Exception as e:
                print(f"ERRORE: {e}")
            
    print("\n" + "="*50)
    print("🏆 CLASSIFICA FINALE")
    print("="*50)
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, wins) in enumerate(sorted_res):
        medal = "🥇" if rank == 0 else ("🥈" if rank == 1 else ("🥉" if rank == 2 else "  "))
        print(f"{medal} {name:20}: {wins} vittorie")
    print("="*50)

if __name__ == "__main__":
    main()
