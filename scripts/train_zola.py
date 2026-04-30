import sys
import time
import random
import json
import os
from datetime import datetime

# Aggiunge la cartella src al path per poter importare i moduli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ZolaGameS import ZolaGame
import playerAdvanced as ai

# Configurazioni Training
GENERATIONS = 30
POPULATION_SIZE = 10
TIMEOUT_PER_MOVE = 1 # Usiamo 1 secondo per mossa per velocizzare (o 0.5)

def generate_random_weights():
    weights = [0] * 68
    # Pesi per quando giochiamo per Primi (0-33)
    for i in range(1, 10): weights[i] = random.randint(-30, 40) # Early
    for i in range(11, 20): weights[i] = random.randint(-30, 40) # Mid
    for i in range(21, 30): weights[i] = random.randint(-30, 40) # Late
    weights[31] = random.randint(-10, 30) # Clustering
    weights[33] = random.randint(500, 1500) # Material
    
    # Pesi per quando giochiamo per Secondi (34-67)
    for i in range(35, 44): weights[i] = random.randint(-30, 40) # Early
    for i in range(45, 54): weights[i] = random.randint(-30, 40) # Mid
    for i in range(55, 64): weights[i] = random.randint(-30, 40) # Late
    weights[65] = random.randint(-10, 30) # Clustering
    weights[67] = random.randint(500, 1500) # Material
    
    return weights

def mutate(weights):
    new_w = list(weights)
    # Mutazioni Primo Giocatore
    for i in list(range(1, 10)) + list(range(11, 20)) + list(range(21, 30)):
        if random.random() < 0.2:
            new_w[i] += random.randint(-10, 10)
    if random.random() < 0.2: new_w[31] += random.randint(-5, 5)
    if random.random() < 0.2: new_w[33] += random.randint(-100, 100)
    
    # Mutazioni Secondo Giocatore
    for i in list(range(35, 44)) + list(range(45, 54)) + list(range(55, 64)):
        if random.random() < 0.2:
            new_w[i] += random.randint(-10, 10)
    if random.random() < 0.2: new_w[65] += random.randint(-5, 5)
    if random.random() < 0.2: new_w[67] += random.randint(-100, 100)
    
    return new_w

import playerExampleAlpha as alpha_bot

def worker_task(args):
    type_match, i, j, w_red, w_blue = args
    
    ai.TT.clear() 
    
    game = ZolaGame()
    state = game.initial
    # I pesi non vengono più splittati qui, ma caricati in BOT_WEIGHTS al momento della mossa.
    
    turns = 0
    while not game.is_terminal(state):
        if turns > 200: return (type_match, i, j, 0)
        
        if state.to_move == "Red":
            if type_match == 'alpha_blue':
                # Alpha gioca come Rosso (Primo)
                move = alpha_bot.playerStrategy(game, state, timeout=TIMEOUT_PER_MOVE)
            else:
                # Noi giochiamo come Rosso (Primo) sia nel RR che in alpha_red
                # Assegnamo il genoma di w_red al bot prima che calcoli la mossa!
                ai.BOT_WEIGHTS = w_red
                move = ai.playerStrategy(game, state, timeout=TIMEOUT_PER_MOVE)
        else:
            if type_match == 'alpha_red':
                # Alpha gioca come Blu (Secondo)
                move = alpha_bot.playerStrategy(game, state, timeout=TIMEOUT_PER_MOVE)
            else:
                # Noi giochiamo come Blu (Secondo) sia nel RR che in alpha_blue
                # Se è RR usiamo w_blue, se è alpha_blue usiamo w_red perché noi siamo il Blu!
                ai.BOT_WEIGHTS = w_blue if type_match == 'rr' else w_red
                move = ai.playerStrategy(game, state, timeout=TIMEOUT_PER_MOVE)
                
        if move == "PASS" or move is None:
            state = game.pass_turn(state)
        else:
            state = game.result(state, move)
        turns += 1
        
    winner = game.winner(state)
    if winner == "Red": return (type_match, i, j, 1)
    elif winner == "Blue": return (type_match, i, j, -1)
    return (type_match, i, j, 0)

def save_state(gen, population, archive):
    state_file = os.path.join(os.path.dirname(__file__), '../logs/training_state.json')
    with open(state_file, "w") as f:
        json.dump({
            "generation": gen,
            "population": population,
            "archive": archive
        }, f, indent=4)

def load_state():
    state_file = os.path.join(os.path.dirname(__file__), '../logs/training_state.json')
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                data = json.load(f)
                return data.get("generation", 0), data.get("population", []), data.get("archive", [])
        except:
            pass
    return 0, [], []

def run_training():
    print("Inizio Training Autonomo (Algoritmo Genetico / Hill Climbing)")
    
    time_limit_str = input("Quanti minuti vuoi che il training giri? (0 per infinito) [default 0]: ")
    try:
        max_minutes = float(time_limit_str)
    except ValueError:
        max_minutes = 0.0
        
    start_time = time.time()
    start_gen, population, archive = load_state()
    
    if len(population) != POPULATION_SIZE:
        print("Nessuno stato precedente compatibile trovato. Inizializzo da zero...")
        population = [generate_random_weights() for _ in range(POPULATION_SIZE)]
        population[0] = ai.DEFAULT_WEIGHTS.copy()
        archive = []
        start_gen = 0
    else:
        print(f"Stato trovato! Riprendo il training dalla Generazione {start_gen+1}...")

    print(f"Popolazione: {POPULATION_SIZE}, Generazioni previste totali: {GENERATIONS}")
    
    gen = start_gen
    try:
        for gen in range(start_gen, GENERATIONS):
            if max_minutes > 0 and (time.time() - start_time) / 60.0 > max_minutes:
                print("\n[Timer Scaduto] Tempo limite raggiunto! Salvo lo stato e mi fermo...")
                save_state(gen, population, archive)
                break
            print(f"\n=== Generazione {gen+1}/{GENERATIONS} ===")
            scores = [0] * POPULATION_SIZE
            # Costruiamo il pool di partite per il multi-processing
            import concurrent.futures
            import multiprocessing
            
            match_tasks = []
            for i in range(POPULATION_SIZE):
                for j in range(POPULATION_SIZE):
                    if i != j:
                        match_tasks.append(('rr', i, j, population[i], population[j]))
                # Il nostro bot sfida Alpha come Rosso (Primo) e come Blu (Secondo)
                match_tasks.append(('alpha_red', i, None, population[i], None))
                match_tasks.append(('alpha_blue', i, None, population[i], None))
                
            total_matches = len(match_tasks)
            matches_done = 0
            max_w = max(1, multiprocessing.cpu_count() - 1)
            print(f"Lancio {total_matches} partite in parallelo sfruttando {max_w} core...")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as executor:
                futures = [executor.submit(worker_task, t) for t in match_tasks]
                for future in concurrent.futures.as_completed(futures):
                    type_match, i, j, res = future.result()
                    matches_done += 1
                    sys.stdout.write(f"\rProgresso: [{matches_done}/{total_matches}] partite completate...")
                    sys.stdout.flush()
                    
                    if type_match == 'rr':
                        if res == 1:
                            scores[i] += 3
                        elif res == -1:
                            scores[j] += 3
                        else:
                            scores[i] += 1
                            scores[j] += 1
                    elif type_match == 'alpha_red':
                        # Noi eravamo Red. Se vince Red (1), vinciamo noi.
                        if res == 1:
                            scores[i] += 5
                        elif res == 0:
                            scores[i] += 2
                    elif type_match == 'alpha_blue':
                        # Noi eravamo Blue. Se vince Blue (-1), vinciamo noi.
                        if res == -1:
                            scores[i] += 5
                        elif res == 0:
                            scores[i] += 2
            print() # A capo dopo la progress bar
            # Ranking
            ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
            print(f"Miglior punteggio: {ranked[0][0]}")
            print(f"Migliori pesi: {ranked[0][1]}")
            
            # Selezione dei migliori (Elitismo)
            best_2 = [ranked[0][1], ranked[1][1]]
            archive.append(ranked[0][1]) # Salviamo il campione nel pool storico
            
            # Creazione nuova generazione
            new_population = [best_2[0], best_2[1]] # I migliori passano intatti
            
            # Iniettiamo un campione del passato per combattere il Catastrophic Forgetting
            if len(archive) > 1 and len(new_population) < POPULATION_SIZE:
                new_population.append(random.choice(archive))
                
            while len(new_population) < POPULATION_SIZE:
                parent = random.choice(best_2)
                child = mutate(parent)
                new_population.append(child)
                
            population = new_population
            
            # Salva lo stato della popolazione corrente (Checkpoint)
            save_state(gen + 1, population, archive)
            
            # Salva backup del migliore finora
            out_file = os.path.join(os.path.dirname(__file__), '../logs/best_weights.json')
            with open(out_file, "w") as f:
                json.dump({
                    "generation": gen + 1,
                    "best_weights": ranked[0][1],
                    "score": ranked[0][0]
                }, f, indent=4)
                
            # Salva storico per la Dashboard (CSV)
            csv_file = os.path.join(os.path.dirname(__file__), '../logs/training_history.csv')
            file_exists = os.path.exists(csv_file)
            with open(csv_file, 'a') as f:
                if not file_exists:
                    f.write("Generation,Score,Material_First,Material_Second,Clustering_First,Clustering_Second\n")
                best_w = ranked[0][1]
                f.write(f"{gen + 1},{ranked[0][0]},{best_w[33]},{best_w[67]},{best_w[31]},{best_w[65]}\n")

            
        print("\nTraining Completato o Fermato! I pesi migliori sono stati salvati in 'logs/best_weights.json'")

    except KeyboardInterrupt:
        print("\n\n[Interrotto dall'utente] Hai premuto Ctrl+C!")
        print("Salvo lo stato attuale della popolazione per poter riprendere in futuro...")
        # Nota: 'gen' è accessibile qui se l'eccezione avviene nel loop
        try:
            save_state(gen, population, archive)
        except:
            pass
        sys.exit(0)

if __name__ == "__main__":
    run_training()
