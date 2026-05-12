import sys
import time
import random
import json
import os
import concurrent.futures
import multiprocessing

# Gestione percorsi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
sys.path.append(SRC_DIR)

from ZolaGameS import ZolaGame
import playerPro130 as ai
import playerExampleAlpha as alpha_bot

# CONFIGURAZIONE
POPULATION_SIZE = 24
GENERATIONS = 100
MAX_DEPTH = 5

def mutate(weights):
    new_w = list(weights)
    for i in range(len(new_w)):
        if random.random() < 0.35:
            new_w[i] += random.uniform(-60, 60)
    return new_w

def crossover(w1, w2):
    return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(w1, w2)]

def calculate_hash(rb, bb, is_red):
    """Calcola l'hash Zobrist reale per la Transposition Table"""
    h = 0
    for idx in range(64):
        if rb & (1 << idx): h ^= ai.ZOBRIST_RED[idx]
        if bb & (1 << idx): h ^= ai.ZOBRIST_BLUE[idx]
    if not is_red: h ^= ai.ZOBRIST_TURN
    return h

def safe_pvs_search(rb, bb, is_red, score, limit=0.8):
    best_m = None
    start_t = time.time()
    h = calculate_hash(rb, bb, is_red)
    
    legal = ai.generate_moves(rb, bb, is_red)
    if not legal: return None
    
    for d in range(1, MAX_DEPTH + 1):
        try:
            remaining = limit - (time.time() - start_t)
            if remaining < 0.05: break
            
            # Passiamo l'hash reale!
            _, m = ai.pvs_search(rb, bb, is_red, d, -1e9, 1e9, h, score, time.time(), remaining, 0)
            if m: best_m = m
        except:
            break
            
    if best_m:
        return ((best_m[0]//8, best_m[0]%8), (best_m[1]//8, best_m[1]%8), best_m[2])
    return random.choice(legal)

def worker_task(args):
    type_match, i, j, w_red, w_blue = args
    try:
        ai.TT.clear()
        game = ZolaGame()
        state = game.initial
        turns = 0
        
        while not game.is_terminal(state) and turns < 150:
            legal = game.actions(state)
            if not legal:
                state = game.pass_turn(state)
                turns += 1
                continue

            if turns < 4 and random.random() < 0.5:
                move = random.choice(legal)
            else:
                cur = state.to_move
                if cur == "Red":
                    if type_match == 'alpha_blue':
                        move = alpha_bot.playerStrategy(game, state, timeout=0.1)
                    else:
                        ai.BOT_WEIGHTS = w_red
                        rb, bb, is_red, score = ai.parse_state(state)
                        move = safe_pvs_search(rb, bb, is_red, score)
                else:
                    if type_match == 'alpha_red':
                        move = alpha_bot.playerStrategy(game, state, timeout=0.1)
                    else:
                        ai.BOT_WEIGHTS = w_blue if type_match == 'rr' else w_red
                        rb, bb, is_red, score = ai.parse_state(state)
                        move = safe_pvs_search(rb, bb, is_red, score)
            
            state = game.result(state, move)
            turns += 1
            
        winner = game.winner(state)
        return (type_match, i, j, 1 if winner == "Red" else (-1 if winner == "Blue" else 0))
    except:
        return (type_match, i, j, 0)

def run_training():
    state_file = os.path.join(LOGS_DIR, 'state_pro.json')
    best_file = os.path.join(LOGS_DIR, 'best_pro_130.json')
    
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            data = json.load(f)
            population = data['pop']
            start_gen = data['gen']
    else:
        population = [ai.BOT_WEIGHTS.copy() for _ in range(POPULATION_SIZE)]
        start_gen = 0

    print(f"🚀 TRAINING REALE (Gen {start_gen}) - ZOBRIST HASH ENABLED")
    cpus = max(1, multiprocessing.cpu_count() - 1)

    for gen in range(start_gen, GENERATIONS):
        tasks = []
        for i in range(POPULATION_SIZE):
            for j in range(POPULATION_SIZE):
                if i != j: tasks.append(('rr', i, j, population[i], population[j]))
            tasks.append(('alpha_red', i, None, population[i], None))
        
        scores = [0] * POPULATION_SIZE
        start_t = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
            futures = [executor.submit(worker_task, t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                m_type, i, j, res = f.result()
                if m_type == 'rr':
                    if res == 1: scores[i]+=3
                    elif res == -1: scores[j]+=3
                    else: scores[i]+=1; scores[j]+=1
                else:
                    if res == 1: scores[i]+=5

        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        print(f"--- Gen {gen+1} | Top Score: {ranked[0][0]} | Tempo: {time.time()-start_t:.1f}s ---")
        
        parents = [ranked[0][1], ranked[1][1], ranked[2][1], ranked[3][1]]
        new_pop = list(parents)
        while len(new_pop) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            new_pop.append(mutate(crossover(p1, p2)))
        
        population = new_pop
        with open(state_file, 'w') as f:
            json.dump({'gen': gen+1, 'pop': population}, f)
        with open(best_file, 'w') as f:
            json.dump({'gen': gen+1, 'weights': ranked[0][1]}, f)

if __name__ == "__main__":
    run_training()
