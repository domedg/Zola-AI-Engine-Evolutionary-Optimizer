import time
import math
import random

# ==========================================
# PRECOMPUTAZIONE E COSTANTI (BITBOARDS)
# ==========================================
SIZE = 8

LEVELS = [
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 6, 6, 6, 6, 6, 6, 7,
    7, 6, 5, 5, 5, 5, 6, 7,
    7, 6, 5, 4, 4, 5, 6, 7,
    7, 6, 5, 4, 4, 5, 6, 7,
    7, 6, 5, 5, 5, 5, 6, 7,
    7, 6, 6, 6, 6, 6, 6, 7,
    7, 7, 7, 7, 7, 7, 7, 7
]

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

def _compute_adj_and_rays():
    adj = [[] for _ in range(64)]
    rays = [[[] for _ in range(8)] for _ in range(64)]
    
    for r in range(SIZE):
        for c in range(SIZE):
            sq = r * SIZE + c
            # Adiacenze per mosse NON catturanti (livello deve essere strettamente maggiore)
            for d_idx, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE:
                    nsq = nr * SIZE + nc
                    if LEVELS[nsq] > LEVELS[sq]:
                        adj[sq].append(nsq)
                
                # Raggi per mosse CATTURANTI (si fermano al primo ostacolo)
                curr_r, curr_c = r + dr, c + dc
                while 0 <= curr_r < SIZE and 0 <= curr_c < SIZE:
                    rays[sq][d_idx].append(curr_r * SIZE + curr_c)
                    curr_r += dr
                    curr_c += dc
    return adj, rays

ADJ_HIGHER, RAYS = _compute_adj_and_rays()

BIT_MASKS = [1 << i for i in range(64)]

# Zobrist Hashing per la Transposition Table
random.seed(42)
ZOBRIST_RED = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_BLUE = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_TURN = random.getrandbits(64)

# Transposition Table globale
TT = {}

# ==========================================
# 130 PARAMETER WEIGHTS
# ==========================================
# 0..63: Red Positional Weights
# 64..127: Blue Positional Weights
# 128: Red Clustering Weight
# 129: Blue Clustering Weight

PESI_ROSSO = [0.0] * 64
PESI_BLU = [0.0] * 64
PESO_CLUSTERING_ROSSO = 0.0
PESO_CLUSTERING_BLU = 0.0

def load_weights():
    global PESI_ROSSO, PESI_BLU, PESO_CLUSTERING_ROSSO, PESO_CLUSTERING_BLU
    try:
        import json
        import os
        weights_path = os.path.join(os.path.dirname(__file__), '../logs/best_weights_130.json')
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                data = json.load(f)
                weights = data.get("weights")
                if weights and len(weights) == 130:
                    PESI_ROSSO = weights[0:64]
                    PESI_BLU = weights[64:128]
                    PESO_CLUSTERING_ROSSO = weights[128]
                    PESO_CLUSTERING_BLU = weights[129]
    except Exception:
        pass

load_weights()

# ==========================================
# FUNZIONI DI SUPPORTO BITBOARD
# ==========================================
def sq_to_rc(sq):
    return (sq // SIZE, sq % SIZE)

def rc_to_sq(r, c):
    return r * SIZE + c

def parse_state(state):
    red_bb = 0
    blue_bb = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if state.board[r][c] == "Red":
                red_bb |= BIT_MASKS[rc_to_sq(r, c)]
            elif state.board[r][c] == "Blue":
                blue_bb |= BIT_MASKS[rc_to_sq(r, c)]
    is_red_turn = (state.to_move == "Red")
    return red_bb, blue_bb, is_red_turn

def get_initial_hash(red_bb, blue_bb, is_red_turn):
    h = 0
    r = red_bb
    while r:
        lsb = r & -r
        sq = lsb.bit_length() - 1
        h ^= ZOBRIST_RED[sq]
        r ^= lsb
    b = blue_bb
    while b:
        lsb = b & -b
        sq = lsb.bit_length() - 1
        h ^= ZOBRIST_BLUE[sq]
        b ^= lsb
    if not is_red_turn:
        h ^= ZOBRIST_TURN
    return h

def generate_moves(red_bb, blue_bb, is_red_turn):
    moves = []
    my_bb = red_bb if is_red_turn else blue_bb
    opp_bb = blue_bb if is_red_turn else red_bb
    occ_bb = red_bb | blue_bb
    
    bb = my_bb
    while bb:
        lsb = bb & -bb
        sq = lsb.bit_length() - 1
        bb ^= lsb
        for target in ADJ_HIGHER[sq]:
            if not (occ_bb & BIT_MASKS[target]):
                moves.append((sq, target, False))
        sq_lvl = LEVELS[sq]
        for d_idx in range(8):
            for target in RAYS[sq][d_idx]:
                mask = BIT_MASKS[target]
                if occ_bb & mask:
                    if opp_bb & mask:
                        if LEVELS[target] <= sq_lvl:
                            moves.append((sq, target, True))
                    break
    return moves

def make_move(red_bb, blue_bb, is_red_turn, hash_val, move):
    if move is None:
        return red_bb, blue_bb, not is_red_turn, hash_val ^ ZOBRIST_TURN
    fr, to, is_cap = move
    fr_mask = BIT_MASKS[fr]
    to_mask = BIT_MASKS[to]
    new_red = red_bb
    new_blue = blue_bb
    new_hash = hash_val ^ ZOBRIST_TURN
    if is_red_turn:
        new_red ^= (fr_mask | to_mask)
        new_hash ^= ZOBRIST_RED[fr] ^ ZOBRIST_RED[to]
        if is_cap:
            new_blue ^= to_mask
            new_hash ^= ZOBRIST_BLUE[to]
    else:
        new_blue ^= (fr_mask | to_mask)
        new_hash ^= ZOBRIST_BLUE[fr] ^ ZOBRIST_BLUE[to]
        if is_cap:
            new_red ^= to_mask
            new_hash ^= ZOBRIST_RED[to]
    return new_red, new_blue, not is_red_turn, new_hash

def evaluate(red_bb, blue_bb, is_red_turn):
    red_count = red_bb.bit_count()
    blue_count = blue_bb.bit_count()
    if blue_count == 0: return 1000000
    if red_count == 0: return -1000000
    
    score = 0.0
    
    # Positional Weights
    r_bb = red_bb
    while r_bb:
        lsb = r_bb & -r_bb
        sq = lsb.bit_length() - 1
        r_bb ^= lsb
        score += PESI_ROSSO[sq]
        
    b_bb = blue_bb
    while b_bb:
        lsb = b_bb & -b_bb
        sq = lsb.bit_length() - 1
        b_bb ^= lsb
        score -= PESI_BLU[sq]
        
    # Clustering
    red_cluster = (red_bb & (red_bb << 1)).bit_count() + (red_bb & (red_bb >> 1)).bit_count() + \
                  (red_bb & (red_bb << 8)).bit_count() + (red_bb & (red_bb >> 8)).bit_count()
    blue_cluster = (blue_bb & (blue_bb << 1)).bit_count() + (blue_bb & (blue_bb >> 1)).bit_count() + \
                   (blue_bb & (blue_bb << 8)).bit_count() + (blue_bb & (blue_bb >> 8)).bit_count()
                   
    score += red_cluster * PESO_CLUSTERING_ROSSO
    score -= blue_cluster * PESO_CLUSTERING_BLU
    
    return score if is_red_turn else -score

def negamax(red_bb, blue_bb, is_red_turn, depth, alpha, beta, hash_val, start_time, time_limit):
    if time.time() - start_time > time_limit:
        raise TimeoutError()
    if red_bb == 0:
        return -1000000 + depth, None
    if blue_bb == 0:
        return 1000000 - depth, None
    tt_entry = TT.get(hash_val)
    if tt_entry and tt_entry['depth'] >= depth:
        flag = tt_entry['flag']
        value = tt_entry['value']
        if flag == 0: return value, tt_entry['best_move']
        elif flag == 1: alpha = max(alpha, value)
        elif flag == 2: beta = min(beta, value)
        if alpha >= beta: return value, tt_entry['best_move']
    if depth == 0:
        return evaluate(red_bb, blue_bb, is_red_turn), None
    moves = generate_moves(red_bb, blue_bb, is_red_turn)
    if not moves:
        n_red, n_blue, n_turn, n_hash = make_move(red_bb, blue_bb, is_red_turn, hash_val, None)
        val, _ = negamax(n_red, n_blue, n_turn, depth - 1, -beta, -alpha, n_hash, start_time, time_limit)
        return -val, None
    tt_best = tt_entry['best_move'] if tt_entry else None
    moves.sort(key=lambda m: (m == tt_best, m[2]), reverse=True)
    best_move = None
    orig_alpha = alpha
    best_val = -math.inf
    for move in moves:
        n_red, n_blue, n_turn, n_hash = make_move(red_bb, blue_bb, is_red_turn, hash_val, move)
        val, _ = negamax(n_red, n_blue, n_turn, depth - 1, -beta, -alpha, n_hash, start_time, time_limit)
        val = -val
        if val > best_val:
            best_val = val
            best_move = move
        alpha = max(alpha, val)
        if alpha >= beta: break
    flag = 0
    if best_val <= orig_alpha: flag = 2
    elif best_val >= beta: flag = 1
    TT[hash_val] = {'depth': depth, 'flag': flag, 'value': best_val, 'best_move': best_move}
    return best_val, best_move

def playerStrategy(game, state, timeout=3):
    global TT
    if len(TT) > 1000000: TT.clear()
    start_time = time.time()
    time_limit = timeout - 0.2
    red_bb, blue_bb, is_red_turn = parse_state(state)
    hash_val = get_initial_hash(red_bb, blue_bb, is_red_turn)
    legal_moves = generate_moves(red_bb, blue_bb, is_red_turn)
    if not legal_moves: return None
    best_move = None
    depth = 1
    while True:
        try:
            val, move = negamax(red_bb, blue_bb, is_red_turn, depth, -math.inf, math.inf, hash_val, start_time, time_limit)
            if move: best_move = move
            depth += 1
        except TimeoutError: break
    if not best_move: best_move = random.choice(legal_moves)
    fr, to, is_cap = best_move
    return (sq_to_rc(fr), sq_to_rc(to), is_cap)
