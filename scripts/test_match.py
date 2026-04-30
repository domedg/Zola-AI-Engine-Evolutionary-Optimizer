import sys
from ZolaGameS import ZolaGame
import playerExampleAlpha as alpha
import playerAdvanced as adv

def run_match():
    game = ZolaGame()
    state = game.initial
    
    # Red is advanced, Blue is alpha
    while not game.is_terminal(state):
        if state.to_move == "Red":
            move = adv.playerStrategy(game, state, timeout=3)
        else:
            move = alpha.playerStrategy(game, state, timeout=3)
            
        if move == "PASS" or move is None:
            state = game.pass_turn(state)
            print(f"{state.to_move} passed.")
        else:
            print(f"{state.to_move} moving: {move}")
            try:
                state = game.result(state, move)
            except Exception as e:
                print(f"Error on move {move}: {e}")
                break
                
        # print counts
        r = state.count("Red")
        b = state.count("Blue")
        print(f"Scores -> Red: {r}, Blue: {b}")
        
    winner = game.winner(state)
    print(f"Game Over! Winner: {winner}")

if __name__ == "__main__":
    run_match()
