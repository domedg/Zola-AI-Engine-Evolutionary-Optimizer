import sys
import os
import random
import time
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ZolaGameS import ZolaGame
import playerGPU as gpu_ai
import playerAdvanced as cpu_ai
from playerAdvanced import parse_state, generate_moves, make_move, sq_to_rc

# ==========================================
# GENETIC ALGORITHM CONFIGURATION
# ==========================================
POPULATION_SIZE = 20
GENERATIONS = 50
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.1
ELITISM_COUNT = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Population:
    def __init__(self, size):
        self.size = size
        # 130 weights per individual (64 Red pos, 64 Blue pos, 1 Red cluster, 1 Blue cluster)
        self.individuals = torch.randn((size, 130), device=DEVICE) * 0.1
        self.scores = np.zeros(size)

    def mutate(self):
        mask = torch.rand(self.individuals.shape, device=DEVICE) < MUTATION_RATE
        noise = torch.randn(self.individuals.shape, device=DEVICE) * MUTATION_STRENGTH
        self.individuals[mask] += noise[mask]

def batched_play_game(population, num_games_per_individual=2):
    """
    Simula partite in parallelo sulla GPU.
    Ogni individuo della popolazione gioca contro il 'CPU Bot' (playerAdvanced).
    """
    pop_size = population.size
    total_sim_games = pop_size * num_games_per_individual
    
    # Stati dei giochi attivi
    active_games = []
    for i in range(pop_size):
        for _ in range(num_games_per_individual):
            active_games.append({
                'game': ZolaGame(),
                'state': ZolaGame().initial,
                'ind_idx': i,
                'gpu_is_red': random.choice([True, False]),
                'finished': False,
                'winner': None,
                'turns': 0
            })
            
    while True:
        # Trova giochi non finiti
        pending = [g for g in active_games if not g['finished']]
        if not pending:
            break
            
        # Per i turni della GPU, facciamo una valutazione batch
        gpu_turn_games = []
        cpu_turn_games = []
        
        for g in pending:
            g['turns'] += 1
            if g['turns'] > 200:
                g['finished'] = True
                continue
                
            red_bb, blue_bb, is_red_turn = parse_state(g['state'])
            if (is_red_turn and g['gpu_is_red']) or (not is_red_turn and not g['gpu_is_red']):
                gpu_turn_games.append(g)
            else:
                cpu_turn_games.append(g)
        
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as cpu_executor:
        while True:
            # Trova giochi non finiti
            pending = [g for g in active_games if not g['finished']]
            if not pending:
                break
                
            # Per i turni della GPU, facciamo una valutazione batch
            gpu_turn_games = []
            cpu_turn_games = []
            
            for g in pending:
                g['turns'] += 1
                if g['turns'] > 200:
                    g['finished'] = True
                    continue
                    
                red_bb, blue_bb, is_red_turn = parse_state(g['state'])
                if (is_red_turn and g['gpu_is_red']) or (not is_red_turn and not g['gpu_is_red']):
                    gpu_turn_games.append(g)
                else:
                    cpu_turn_games.append(g)
            
            # 1. Gestione Turni CPU (Advanced Bot) - Parallelizzato
            if cpu_turn_games:
                future_to_game = {cpu_executor.submit(cpu_ai.playerStrategy, g['game'], g['state'], 0.1): g for g in cpu_turn_games}
                for future in concurrent.futures.as_completed(future_to_game):
                    g = future_to_game[future]
                    try:
                        move = future.result()
                    except Exception as e:
                        print(f"Errore CPU Bot: {e}")
                        move = None
                        
                    if move is None or move == "PASS":
                        g['state'] = g['game'].pass_turn(g['state'])
                    else:
                        g['state'] = g['game'].result(g['state'], move)
                    
                    if g['game'].is_terminal(g['state']):
                        g['finished'] = True
                        g['winner'] = g['game'].winner(g['state'])

        # 2. Gestione Turni GPU (Batched)
        if gpu_turn_games:
            all_moves = []
            all_state_tensors = []
            game_to_move_map = []
            
            for g in gpu_turn_games:
                r_bb, b_bb, is_red = parse_state(g['state'])
                moves = generate_moves(r_bb, b_bb, is_red)
                if not moves:
                    g['state'] = g['game'].pass_turn(g['state'])
                    if g['game'].is_terminal(g['state']):
                        g['finished'] = True
                        g['winner'] = g['game'].winner(g['state'])
                    continue
                
                # Per ogni mossa legale, calcola il tensore dello stato risultante
                my_bbs = []
                opp_bbs = []
                for m in moves:
                    nr, nb, nt, _ = make_move(r_bb, b_bb, is_red, 0, m)
                    my_bbs.append(nr)
                    opp_bbs.append(nb)
                
                tensors = gpu_ai.bitboards_to_tensor(my_bbs, opp_bbs) # (NumMoves, 130)
                
                all_state_tensors.append(tensors)
                all_moves.append(moves)
                game_to_move_map.append(g)
            
            if all_state_tensors:
                # Concateniamo tutti i tensori di tutti i giochi per un'unica passata GPU
                batch_tensors = torch.cat(all_state_tensors, dim=0) # (TotalMoves, 130)
                
                # Recuperiamo i pesi degli individui corrispondenti
                # Poiché ogni mossa appartiene a un gioco che appartiene a un individuo
                weights_list = []
                for idx, g in enumerate(game_to_move_map):
                    num_m = len(all_moves[idx])
                    w = population.individuals[g['ind_idx']].unsqueeze(0) # (1, 130)
                    weights_list.append(w.expand(num_m, -1))
                
                batch_weights = torch.cat(weights_list, dim=0) # (TotalMoves, 130)
                
                # Valutazione Lineare: Dot Product (Element-wise multiply + sum)
                # score = sum(features * weights)
                scores = (batch_tensors * batch_weights).sum(dim=1)
                
                # Distribuiamo gli score ai rispettivi giochi per scegliere la mossa migliore
                offset = 0
                for idx, g in enumerate(game_to_move_map):
                    num_m = len(all_moves[idx])
                    game_scores = scores[offset:offset+num_m]
                    offset += num_m
                    
                    # Se è il turno del Blu per la GPU, vogliamo minimizzare (o invertire score se il modello è per il Rosso)
                    # Ma qui il modello calcola direttamente il 'vantaggio' del giocatore corrente.
                    # Per coerenza con Zola, facciamo ArgMax degli score.
                    best_idx = torch.argmax(game_scores).item()
                    move = all_moves[idx][best_idx]
                    
                    # Esegui mossa
                    fr_sq, to_sq, is_cap = move
                    move_rc = (sq_to_rc(fr_sq), sq_to_rc(to_sq), is_cap)
                    g['state'] = g['game'].result(g['state'], move_rc)
                    
                    if g['game'].is_terminal(g['state']):
                        g['finished'] = True
                        g['winner'] = g['game'].winner(g['state'])

    # Calcolo punteggi finali
    for g in active_games:
        win_val = 0
        if g['winner'] == "Red":
            win_val = 1 if g['gpu_is_red'] else -1
        elif g['winner'] == "Blue":
            win_val = 1 if not g['gpu_is_red'] else -1
            
        if win_val == 1:
            population.scores[g['ind_idx']] += 3
        elif win_val == 0:
            population.scores[g['ind_idx']] += 1

def convert_68_to_130(w68):
    """Converte i pesi del bot a 68 parametri nel formato a 130 parametri."""
    w130 = torch.zeros(130, device=DEVICE)
    # Pesi Rossi (Primi)
    for sq in range(64):
        lvl = cpu_ai.LEVELS[sq]
        # Usiamo la fase 'Mid' (offset 10) come base per i pesi posizionali
        w130[sq] = float(w68[10 + lvl]) + (float(w68[33]) / 10.0) # Aggiungiamo un decimo del peso materiale
    
    # Pesi Blu (Secondi)
    for sq in range(64):
        lvl = cpu_ai.LEVELS[sq]
        w130[64 + sq] = float(w68[34 + 10 + lvl]) + (float(w68[67]) / 10.0)
        
    w130[128] = float(w68[31]) # Clustering Rosso
    w130[129] = float(w68[65]) # Clustering Blu
    return w130

def run_genetic_training():
    log_file = "logs/genetic_training.log"
    os.makedirs("logs", exist_ok=True)
    
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    log(f"Inizio Genetic GPU Training su {DEVICE}")
    log(f"Popolazione: {POPULATION_SIZE}, Generazioni: {GENERATIONS}")
    
    pop = Population(POPULATION_SIZE)
    
    # INIZIALIZZAZIONE SMART: Usiamo i pesi del bot Advanced come base per i primi 2 individui
    try:
        adv_weights = cpu_ai.BOT_WEIGHTS
        smart_init = convert_68_to_130(adv_weights)
        pop.individuals[0] = smart_init
        pop.individuals[1] = smart_init + (torch.randn(130, device=DEVICE) * 0.05)
        log("Individui 0 e 1 inizializzati con i pesi del Bot Advanced (68 -> 130 conversion).")
    except Exception as e:
        log(f"Errore inizializzazione smart: {e}")

    for gen in range(GENERATIONS):
        start_gen = time.time()
        pop.scores.fill(0)
        
        log(f"\n--- Generazione {gen+1}/{GENERATIONS} ---")
        
        # Simuliamo partite per ogni individuo
        # Aumentiamo a 8 partite per individuo per maggiore stabilità
        batched_play_game(pop, num_games_per_individual=8)
        
        # Ranking
        indices = np.argsort(pop.scores)[::-1].copy()
        best_score = pop.scores[indices[0]]
        log(f"Miglior Punteggio: {best_score} (Media: {np.mean(pop.scores):.1f})")
        
        # Selezione (Elitismo)
        new_individuals = pop.individuals[indices[:ELITISM_COUNT]].clone()
        
        # Cross-over & Mutation
        children = []
        while len(children) < (POPULATION_SIZE - ELITISM_COUNT):
            p1_idx = indices[random.randint(0, 5)] # Seleziona tra i top 6
            p2_idx = indices[random.randint(0, 5)]
            
            # Blend Crossover
            alpha = random.random()
            child = alpha * pop.individuals[p1_idx] + (1 - alpha) * pop.individuals[p2_idx]
            children.append(child)
            
        pop.individuals = torch.cat([new_individuals, torch.stack(children)], dim=0)
        pop.mutate()
        
        # Salva il migliore della generazione
        best_weights = pop.individuals[0]
        gpu_ai.model.fc.weight.data[0] = best_weights
        torch.save(gpu_ai.model.state_dict(), 'logs/gpu_weights.pth')
        
        duration = time.time() - start_gen
        log(f"Generazione completata in {duration:.1f}s")

    log("\nTraining Completato! Migliori pesi salvati in 'logs/gpu_weights.pth'")

if __name__ == "__main__":
    run_genetic_training()
