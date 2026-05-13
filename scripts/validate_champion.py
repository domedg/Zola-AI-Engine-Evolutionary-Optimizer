import sys, os, time, random, copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

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

NUM_GAMES = 20
TIMEOUT = 3.0

def run():
    print("\n" + "="*50)
    print("🏆 VALIDAZIONE CHAMPION VS TT_MO_KM (3 SEC)")
    print("="*50)
    
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
                try:
                    state = game.pass_turn(state)
                except:
                    state.to_move = "Blue" if state.to_move == "Red" else "Red"
                turns += 1
                continue
            
            is_champ_turn = (state.to_move == "Red") == champ_is_red
            strategy = champ.playerStrategy if is_champ_turn else opponent.playerStrategy
            
            # Use a copy to prevent accidental state modification
            state_copy = copy.deepcopy(state)
            try:
                move = strategy(game, state_copy, TIMEOUT)
            except Exception as e:
                # print(f"Error: {e}")
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
        
        print(f"Gara {game_num+1:2}: Champ={champ_color:4} -> {result:4} | Classifica: {wins}W - {losses}L - {draws}D")
    
    wr = (wins / NUM_GAMES) * 100
    print("\n" + "="*50)
    print(f"RISULTATO FINALE: {wins}W - {losses}L - {draws}D")
    print(f"Win Rate: {wr:.1f}%")
    if wr >= 60:
        print("✅ SUCCESSO: Il Champion batte solidamente il benchmark!")
    else:
        print("⚠️  ATTENZIONE: Il Champion non raggiunge ancora il target del 60%.")
    print("="*50)

if __name__ == "__main__":
    run()
