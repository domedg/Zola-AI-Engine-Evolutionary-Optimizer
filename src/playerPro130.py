import time
import math
import random
import os
import json

# ==========================================
# COSTANTI E BITBOARD
# ==========================================
SIZE = 8
BIT_MASKS = [1 << i for i in range(64)]

def _compute_adj_and_rays():
    levels = [0] * 64
    for r in range(SIZE):
        for c in range(SIZE):
            v = (2 * r - (SIZE - 1)) ** 2 + (2 * c - (SIZE - 1)) ** 2
            levels[r * SIZE + c] = v
    unique = sorted(set(levels))
    lvl_map = {v: i + 1 for i, v in enumerate(unique)}
    final_levels = [lvl_map[v] for v in levels]
    
    adj = [[] for _ in range(64)]
    rays = [[[] for _ in range(8)] for _ in range(64)]
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    for r in range(SIZE):
        for c in range(SIZE):
            sq = r * SIZE + c
            for d_idx, (dr, dc) in enumerate(dirs):
                nr, nc = r + dr, c + dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE:
                    nsq = nr * SIZE + nc
                    if final_levels[nsq] > final_levels[sq]: adj[sq].append(nsq)
                curr_r, curr_c = r + dr, c + dc
                while 0 <= curr_r < SIZE and 0 <= curr_c < SIZE:
                    rays[sq][d_idx].append(curr_r * SIZE + curr_c)
                    curr_r += dr; curr_c += dc
    return final_levels, adj, rays

LEVELS, ADJ_HIGHER, RAYS = _compute_adj_and_rays()

# Zobrist Hashing
random.seed(42)
ZOBRIST_RED = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_BLUE = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_TURN = random.getrandbits(64)

# Memorie Globale (Transposition Table)
TT = {}
TT_LIMIT = 5000000 
KILLERS = [[None, None] for _ in range(100)]

# 130 PARAMETRI
BOT_WEIGHTS = [0.0] * 130

def load_weights():
    global BOT_WEIGHTS
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'best_pro_130.json')
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                w = data.get("weights")
                if w and len(w) >= 130: BOT_WEIGHTS[:] = w[:130]
        except: pass

load_weights()

# ==========================================
# MOTORE CORE
# ==========================================
def parse_state(state):
    rb, bb, score = 0, 0, 0.0
    for r in range(SIZE):
        for c in range(SIZE):
            sq = r * SIZE + c
            if state.board[r][c] == "Red":
                rb |= BIT_MASKS[sq]
                score += BOT_WEIGHTS[sq]
            elif state.board[r][c] == "Blue":
                bb |= BIT_MASKS[sq]
                score -= BOT_WEIGHTS[64 + sq]
    rc = (rb & (rb << 1)).bit_count() + (rb & (rb << 8)).bit_count()
    bc = (bb & (bb << 1)).bit_count() + (bb & (bb << 8)).bit_count()
    score += rc * BOT_WEIGHTS[128] - bc * BOT_WEIGHTS[129]
    return rb, bb, (state.to_move == "Red"), score

def make_move_pro(rb, bb, is_red, move, current_score):
    if move is None: return rb, bb, not is_red, -current_score
    fr, to, is_cap = move
    new_rb, new_bb = rb, bb
    new_score = current_score
    if is_red:
        new_rb ^= (BIT_MASKS[fr] | BIT_MASKS[to])
        new_score += (BOT_WEIGHTS[to] - BOT_WEIGHTS[fr])
        if is_cap:
            new_bb ^= BIT_MASKS[to]
            new_score += BOT_WEIGHTS[64 + to]
    else:
        new_bb ^= (BIT_MASKS[fr] | BIT_MASKS[to])
        new_score -= (BOT_WEIGHTS[64 + to] - BOT_WEIGHTS[64 + fr])
        if is_cap:
            new_rb ^= BIT_MASKS[to]
            new_score -= BOT_WEIGHTS[to]
    return new_rb, new_bb, not is_red, -new_score

def generate_moves(rb, bb, is_red):
    moves = []
    my, opp = (rb, bb) if is_red else (bb, rb)
    occ = rb | bb
    temp_my = my
    while temp_my:
        lsb = temp_my & -temp_my
        sq = lsb.bit_length() - 1
        temp_my ^= lsb
        for target in ADJ_HIGHER[sq]:
            if not (occ & BIT_MASKS[target]): moves.append((sq, target, False))
        slvl = LEVELS[sq]
        for d in range(8):
            for target in RAYS[sq][d]:
                if occ & BIT_MASKS[target]:
                    if opp & BIT_MASKS[target] and LEVELS[target] <= slvl: moves.append((sq, target, True))
                    break
    return moves

def pvs_search(rb, bb, is_red, depth, alpha, beta, h, score, start_time, limit, ply):
    if time.time() - start_time > limit: raise TimeoutError()
    
    # Condizioni terminali veloci
    if rb == 0: return -10000000 + ply, None
    if bb == 0: return 10000000 - ply, None

    # TT Probe
    tt_entry = TT.get(h)
    tt_move = None
    if tt_entry and tt_entry['d'] >= depth:
        val = tt_entry['v']
        if tt_entry['f'] == 0: return val, tt_entry['m']
        elif tt_entry['f'] == 1: alpha = max(alpha, val)
        elif tt_entry['f'] == 2: beta = min(beta, val)
        if alpha >= beta: return val, tt_entry['m']
        tt_move = tt_entry['m']

    if depth <= 0: return score, None

    moves = generate_moves(rb, bb, is_red)
    if not moves:
        _, _, nt, nscore = make_move_pro(rb, bb, is_red, None, score)
        v, _ = pvs_search(rb, bb, nt, depth - 1, -beta, -alpha, h ^ ZOBRIST_TURN, nscore, start_time, limit, ply + 1)
        return -v, None

    # Move Ordering
    k1, k2 = KILLERS[ply] if ply < 100 else (None, None)
    moves.sort(key=lambda m: 1000 if m == tt_move else (500 if m[2] else (100 if m == k1 or m == k2 else 0)), reverse=True)

    best_v = -math.inf
    best_m = None
    old_alpha = alpha

    for i, m in enumerate(moves):
        nrb, nbb, nt, nscore = make_move_pro(rb, bb, is_red, m, score)
        nh = h ^ ZOBRIST_TURN ^ (ZOBRIST_RED[m[0]] ^ ZOBRIST_RED[m[1]] if is_red else ZOBRIST_BLUE[m[0]] ^ ZOBRIST_BLUE[m[1]])
        if m[2]: nh ^= (ZOBRIST_BLUE[m[1]] if is_red else ZOBRIST_RED[m[1]])
        
        if i == 0:
            # Ricerca a finestra piena per la mossa principale
            v, _ = pvs_search(nrb, nbb, nt, depth - 1, -beta, -alpha, nh, nscore, start_time, limit, ply + 1)
            v = -v
        else:
            # Ricerca a finestra nulla (Zero Window Search)
            v, _ = pvs_search(nrb, nbb, nt, depth - 1, -alpha - 1, -alpha, nh, nscore, start_time, limit, ply + 1)
            v = -v
            if alpha < v < beta:
                # Ricerca completa se la finestra nulla fallisce
                v, _ = pvs_search(nrb, nbb, nt, depth - 1, -beta, -alpha, nh, nscore, start_time, limit, ply + 1)
                v = -v
        
        if v > best_v:
            best_v = v
            best_m = m
        
        alpha = max(alpha, v)
        if alpha >= beta:
            if not m[2] and ply < 100:
                KILLERS[ply][1] = KILLERS[ply][0]
                KILLERS[ply][0] = m
            break

    # TT Store
    flag = 0 if best_v > old_alpha and best_v < beta else (1 if best_v >= beta else 2)
    if len(TT) < TT_LIMIT:
        TT[h] = {'d': depth, 'v': best_v, 'm': best_m, 'f': flag}
    
    return best_v, best_m

def playerStrategy(game, state, timeout=3):
    global TT, KILLERS
    for k in KILLERS: k[:] = [None, None]
    rb, bb, is_red, score = parse_state(state)
    h = 0
    for i in range(64):
        if rb & BIT_MASKS[i]: h ^= ZOBRIST_RED[i]
        if bb & BIT_MASKS[i]: h ^= ZOBRIST_BLUE[i]
    if not is_red: h ^= ZOBRIST_TURN
    
    start = time.time()
    limit = timeout - 0.15
    best_m, depth = None, 1
    legal = generate_moves(rb, bb, is_red)
    if not legal: return None
    
    while True:
        try:
            v, m = pvs_search(rb, bb, is_red, depth, -math.inf, math.inf, h, score, start, limit, 0)
            if m: best_m = m
            if v > 9000000 or v < -9000000: break
            depth += 1
        except TimeoutError: break
        
    m = best_m or random.choice(legal)
    return (m[0]//8, m[0]%8), (m[1]//8, m[1]%8), m[2]
