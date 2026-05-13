import sys
import time
import random
import json
import os
import concurrent.futures
import multiprocessing

# Aggiunge la cartella src al path
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
import playerChampion as ai
import player_TT_MO_KM as opponent
import playerAdvanced as advanced

# Configurazioni Training
POPULATION_SIZE = 12
GENERATIONS = 100
TIMEOUT_PER_MOVE = 0.1 # Molto veloce per favorire molte generazioni

def mutate(weights):
    new_w = list(weights)
    for i in range(len(new_w)):
        if random.random() < 0.25:
            if i in [33, 67]: # Material weights
                new_w[i] += random.randint(-200, 200)
            else: # Positional and clustering
                new_w[i] += random.randint(-30, 30)
    return new_w

def crossover(w1, w2):
    return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(w1, w2)]

def worker_task(args):
    type_match, i, w = args
    ai.TT.clear()
    game = ZolaGame()
    state = game.initial
    turns = 0
    
    # In questa versione semplificata:
    # type_match 0: Champ vs TT_MO_KM (Champ is Red)
    # type_match 1: TT_MO_KM vs Champ (Champ is Blue)
    # type_match 2: Champ vs Advanced (Champ is Red)
    # type_match 3: Advanced vs Champ (Champ is Blue)
    
    while not game.is_terminal(state) and turns < 150:
        legal = game.actions(state)
        if not legal:
            state = game.pass_turn(state)
            turns += 1
            continue
            
        cur = state.to_move
        if cur == "Red":
            if type_match in [1, 3]: # Opponent is Red
                opp_mod = opponent if type_match == 1 else advanced
                move = opp_mod.playerStrategy(game, state, TIMEOUT_PER_MOVE)
            else: # Champ is Red
                ai.BOT_WEIGHTS = w
                move = ai.playerStrategy(game, state, TIMEOUT_PER_MOVE)
        else:
            if type_match in [0, 2]: # Opponent is Blue
                opp_mod = opponent if type_match == 0 else advanced
                move = opp_mod.playerStrategy(game, state, TIMEOUT_PER_MOVE)
            else: # Champ is Blue
                ai.BOT_WEIGHTS = w
                move = ai.playerStrategy(game, state, TIMEOUT_PER_MOVE)
        
        if move not in legal:
            move = random.choice(legal)
        state = game.result(state, move)
        turns += 1
        
    winner = game.winner(state)
    champ_won = False
    if type_match in [0, 2] and winner == "Red": champ_won = True
    elif type_match in [1, 3] and winner == "Blue": champ_won = True
    
    is_draw = (winner is None)
    
    return i, type_match, 1 if champ_won else (0.5 if is_draw else 0)

def run_training():
    print("=== ZOLA CHAMPION GA TRAINER ===")
    
    best_weights_path = os.path.join(os.path.dirname(__file__), '../logs/best_weights_champion.json')
    
    initial_weights = [0] * 68
    initial_weights[33] = 1000
    initial_weights[67] = 1000
    
    if os.path.exists(best_weights_path):
        with open(best_weights_path, 'r') as f:
            data = json.load(f)
            weights = data.get("weights")
            if weights and len(weights) == 68:
                initial_weights = weights
                print("Loaded initial weights from texel_tune output.")

    population = [initial_weights.copy()]
    while len(population) < POPULATION_SIZE:
        population.append(mutate(initial_weights))
        
    cpus = max(1, multiprocessing.cpu_count() - 1)

    for gen in range(GENERATIONS):
        match_tasks = []
        for i in range(POPULATION_SIZE):
            for t in range(4): # 4 matches per individual
                match_tasks.append((t, i, population[i]))
        
        scores = [0.0] * POPULATION_SIZE
        print(f"Gen {gen+1}/{GENERATIONS} | Running {len(match_tasks)} matches on {cpus} cores...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
            futures = [executor.submit(worker_task, t) for t in match_tasks]
            for f in concurrent.futures.as_completed(futures):
                idx, t_match, res = f.result()
                # Scoring: Win vs TT_MO_KM (0,1) = 10, Win vs Advanced (2,3) = 5, Draw = 1
                if res == 1:
                    scores[idx] += 10 if t_match < 2 else 5
                elif res == 0.5:
                    scores[idx] += 1
        
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        print(f"Gen {gen+1} | Best Score: {ranked[0][0]}")
        
        # Elitism
        new_pop = [ranked[0][1], ranked[1][1], ranked[2][1]]
        
        while len(new_pop) < POPULATION_SIZE:
            if random.random() < 0.3: # Crossover
                p1, p2 = random.sample(new_pop[:3], 2)
                new_pop.append(mutate(crossover(p1, p2)))
            else: # Mutation
                parent = random.choice(new_pop[:3])
                new_pop.append(mutate(parent))
        
        population = new_pop
        
        # Save best weights
        with open(best_weights_path, "w") as f:
            json.dump({"weights": ranked[0][1], "score": ranked[0][0], "gen": gen+1}, f, indent=4)

if __name__ == "__main__":
    run_training()
