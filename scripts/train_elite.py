import sys
import time
import random
import json
import os
from datetime import datetime

# Aggiunge la cartella src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ZolaGameS import ZolaGame
import playerElite as ai # USIAMO IL NUOVO PLAYER ELITE
import playerExampleAlpha as alpha_bot

# Configurazioni Training
GENERATIONS = 50 # Puntiamo più in alto!
POPULATION_SIZE = 10
TIMEOUT_PER_MOVE = 0.8 # Velocizziamo leggermente per fare più partite

def mutate(weights):
    new_w = list(weights)
    for i in list(range(1, 10)) + list(range(11, 20)) + list(range(21, 30)) + \
             list(range(35, 44)) + list(range(45, 54)) + list(range(55, 64)):
        if random.random() < 0.2:
            new_w[i] += random.randint(-15, 15) # Mutazione un po' più aggressiva
    if random.random() < 0.2: new_w[31] += random.randint(-5, 5) # Clustering Rosso
    if random.random() < 0.2: new_w[65] += random.randint(-5, 5) # Clustering Blu
    if random.random() < 0.2: new_w[33] += random.randint(-150, 150) # Materiale Rosso
    if random.random() < 0.2: new_w[67] += random.randint(-150, 150) # Materiale Blu
    return new_w

def worker_task(args):
    type_match, i, j, w_red, w_blue = args
    ai.TT.clear() 
    game = ZolaGame()
    state = game.initial
    turns = 0
    while not game.is_terminal(state):
        if turns > 200: return (type_match, i, j, 0)
        if state.to_move == "Red":
            if type_match == 'alpha_blue':
                move = alpha_bot.playerStrategy(game, state, timeout=TIMEOUT_PER_MOVE)
            else:
                ai.BOT_WEIGHTS = w_red
                move = ai.playerStrategy(game, state, timeout=TIMEOUT_PER_MOVE)
        else:
            if type_match == 'alpha_red':
                move = alpha_bot.playerStrategy(game, state, timeout=TIMEOUT_PER_MOVE)
            else:
                ai.BOT_WEIGHTS = w_blue if type_match == 'rr' else w_red
                move = ai.playerStrategy(game, state, timeout=TIMEOUT_PER_MOVE)
        if move == "PASS" or move is None: state = game.pass_turn(state)
        else: state = game.result(state, move)
        turns += 1
    winner = game.winner(state)
    if winner == "Red": return (type_match, i, j, 1)
    elif winner == "Blue": return (type_match, i, j, -1)
    return (type_match, i, j, 0)

def save_state(gen, population, archive):
    state_file = os.path.join(os.path.dirname(__file__), '../logs/training_state_elite.json')
    with open(state_file, "w") as f:
        json.dump({"generation": gen, "population": population, "archive": archive}, f, indent=4)

def load_state():
    # Prova a caricare lo stato ELITE, se non esiste prova a caricare quello vecchio per non perdere i progressi!
    state_file = os.path.join(os.path.dirname(__file__), '../logs/training_state_elite.json')
    old_state = os.path.join(os.path.dirname(__file__), '../logs/training_state.json')
    
    target = state_file if os.path.exists(state_file) else old_state
    if os.path.exists(target):
        try:
            with open(target, "r") as f:
                data = json.load(f)
                return data.get("generation", 0), data.get("population", []), data.get("archive", [])
        except: pass
    return 0, [], []

def run_training():
    print("=== ZOLA ELITE EVOLUTIONARY OPTIMIZER ===")
    start_gen, population, archive = load_state()
    
    if not population:
        print("Inizializzazione nuova popolazione...")
        population = [ai.BOT_WEIGHTS.copy() for _ in range(POPULATION_SIZE)]
        start_gen = 0
    else:
        print(f"Stato caricato! Riprendo dalla Generazione {start_gen}...")

    import concurrent.futures
    import multiprocessing
    max_w = max(1, multiprocessing.cpu_count() - 1)

    for gen in range(start_gen, GENERATIONS):
        print(f"\n--- Generazione {gen+1} ---")
        scores = [0] * POPULATION_SIZE
        match_tasks = []
        for i in range(POPULATION_SIZE):
            for j in range(POPULATION_SIZE):
                if i != j: match_tasks.append(('rr', i, j, population[i], population[j]))
            match_tasks.append(('alpha_red', i, None, population[i], None))
            match_tasks.append(('alpha_blue', i, None, population[i], None))

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as executor:
            futures = [executor.submit(worker_task, t) for t in match_tasks]
            done = 0
            for future in concurrent.futures.as_completed(futures):
                type_match, i, j, res = future.result()
                done += 1
                sys.stdout.write(f"\rPartite: {done}/{len(match_tasks)}")
                if type_match == 'rr':
                    if res == 1: scores[i] += 3
                    elif res == -1: scores[j] += 3
                    else: scores[i]+=1; scores[j]+=1
                elif type_match == 'alpha_red' and res == 1: scores[i] += 5
                elif type_match == 'alpha_blue' and res == -1: scores[i] += 5
        
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        print(f"\nBest Score: {ranked[0][0]}")
        
        # Elitismo e Mutazione
        new_pop = [ranked[0][1], ranked[1][1]]
        archive.append(ranked[0][1])
        if len(archive) > 5: archive.pop(0)
        
        while len(new_pop) < POPULATION_SIZE:
            parent = random.choice(new_pop[:2] + ([random.choice(archive)] if archive else []))
            new_pop.append(mutate(parent))
        
        population = new_pop
        save_state(gen + 1, population, archive)
        
        # Salva il migliore assoluto
        with open(os.path.join(os.path.dirname(__file__), '../logs/best_weights_elite.json'), "w") as f:
            json.dump({"generation": gen+1, "weights": ranked[0][1], "score": ranked[0][0]}, f, indent=4)

if __name__ == "__main__":
    run_training()
