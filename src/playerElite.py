import time
import math
import random
import os
import json

# ==========================================
# PRECOMPUTAZIONE E COSTANTI (BITBOARDS)
# ==========================================
SIZE = 8

def _compute_levels():
    scaled_distances = {}
    unique_values = set()
    for r in range(SIZE):
        for c in range(SIZE):
            value = (2 * r - (SIZE - 1)) ** 2 + (2 * c - (SIZE - 1)) ** 2
            scaled_distances[(r, c)] = value
            unique_values.add(value)
    ordered_values = sorted(unique_values)
    level_of = {value: index + 1 for index, value in enumerate(ordered_values)}
    
    levels = [0] * 64
    for r in range(SIZE):
        for c in range(SIZE):
            levels[r * SIZE + c] = level_of[scaled_distances[(r, c)]]
    return levels

LEVELS = _compute_levels()

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
            for d_idx, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < SIZE and 0 <= nc < SIZE:
                    nsq = nr * SIZE + nc
                    if LEVELS[nsq] > LEVELS[sq]:
                        adj[sq].append(nsq)
                
                curr_r, curr_c = r + dr, c + dc
                while 0 <= curr_r < SIZE and 0 <= curr_c < SIZE:
                    rays[sq][d_idx].append(curr_r * SIZE + curr_c)
                    curr_r += dr
                    curr_c += dc
    return adj, rays

ADJ_HIGHER, RAYS = _compute_adj_and_rays()
BIT_MASKS = [1 << i for i in range(64)]

# Zobrist Hashing
random.seed(42)
ZOBRIST_RED = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_BLUE = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_TURN = random.getrandbits(64)

# Memorie di Ricerca
TT = {}
KILLERS = [[None, None] for _ in range(100)] # Killer Moves per profondità

# 68 Parametri (come playerAdvanced)
BOT_WEIGHTS = [0] * 68
ACTIVE_WEIGHTS = BOT_WEIGHTS[0:34]

def load_best_weights():
    global BOT_WEIGHTS
    try:
        path = os.path.join(os.path.dirname(__file__), '../logs/best_weights.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                weights = data.get("best_weights")
                if weights and len(weights) == 68:
                    BOT_WEIGHTS = list(weights)
    except Exception:
        pass

load_best_weights()

# ==========================================
# FUNZIONI DI SUPPORTO BITBOARD
# ==========================================
def sq_to_rc(sq): return (sq // SIZE, sq % SIZE)
def rc_to_sq(r, c): return r * SIZE + c

def parse_state(state):
    red_bb, blue_bb = 0, 0
    for r in range(SIZE):
        for c in range(SIZE):
            if state.board[r][c] == "Red": red_bb |= BIT_MASKS[r * SIZE + c]
            elif state.board[r][c] == "Blue": blue_bb |= BIT_MASKS[r * SIZE + c]
    return red_bb, blue_bb, (state.to_move == "Red")

def get_hash(red_bb, blue_bb, is_red_turn):
    h = 0
    r, b = red_bb, blue_bb
    while r:
        lsb = r & -r
        h ^= ZOBRIST_RED[lsb.bit_length() - 1]
        r ^= lsb
    while b:
        lsb = b & -b
        h ^= ZOBRIST_BLUE[lsb.bit_length() - 1]
        b ^= lsb
    if not is_red_turn: h ^= ZOBRIST_TURN
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
            if not (occ_bb & BIT_MASKS[target]): moves.append((sq, target, False))
        sq_lvl = LEVELS[sq]
        for d_idx in range(8):
            for target in RAYS[sq][d_idx]:
                mask = BIT_MASKS[target]
                if occ_bb & mask:
                    if opp_bb & mask and LEVELS[target] <= sq_lvl: moves.append((sq, target, True))
                    break
    return moves

def make_move(red_bb, blue_bb, is_red_turn, hash_val, move):
    if move is None: return red_bb, blue_bb, not is_red_turn, hash_val ^ ZOBRIST_TURN
    fr, to, is_cap = move
    fr_m, to_m = BIT_MASKS[fr], BIT_MASKS[to]
    new_red, new_blue, new_hash = red_bb, blue_bb, hash_val ^ ZOBRIST_TURN
    if is_red_turn:
        new_red ^= (fr_m | to_m)
        new_hash ^= ZOBRIST_RED[fr] ^ ZOBRIST_RED[to]
        if is_cap:
            new_blue ^= to_m
            new_hash ^= ZOBRIST_BLUE[to]
    else:
        new_blue ^= (fr_m | to_m)
        new_hash ^= ZOBRIST_BLUE[fr] ^ ZOBRIST_BLUE[to]
        if is_cap:
            new_red ^= to_m
            new_hash ^= ZOBRIST_RED[to]
    return new_red, new_blue, not is_red_turn, new_hash

def evaluate(red_bb, blue_bb, is_red_turn):
    rc, bc = red_bb.bit_count(), blue_bb.bit_count()
    if bc == 0: return 10000000
    if rc == 0: return -10000000
    total = rc + bc
    phase_offset = 0 if total > 48 else (10 if total > 24 else 20)
    score = (rc - bc) * ACTIVE_WEIGHTS[33]
    r_bb, b_bb = red_bb, blue_bb
    while r_bb:
        lsb = r_bb & -r_bb
        score += ACTIVE_WEIGHTS[phase_offset + LEVELS[lsb.bit_length() - 1]]
        r_bb ^= lsb
    while b_bb:
        lsb = b_bb & -b_bb
        score -= ACTIVE_WEIGHTS[phase_offset + LEVELS[lsb.bit_length() - 1]]
        b_bb ^= lsb
    red_cl = (red_bb & (red_bb << 1)).bit_count() + (red_bb & (red_bb >> 8)).bit_count()
    blue_cl = (blue_bb & (blue_bb << 1)).bit_count() + (blue_bb & (blue_bb >> 8)).bit_count()
    score += (red_cl - blue_cl) * ACTIVE_WEIGHTS[31]
    return score if is_red_turn else -score

# ==========================================
# MOTORE DI RICERCA ELITE (KM + TT + ORDER)
# ==========================================
def negamax(red_bb, blue_bb, is_red_turn, depth, alpha, beta, hash_val, start_time, time_limit, ply):
    if time.time() - start_time > time_limit: raise TimeoutError()
    if red_bb == 0: return -10000000 + ply, None
    if blue_bb == 0: return 10000000 - ply, None

    tt_entry = TT.get(hash_val)
    tt_move = None
    if tt_entry and tt_entry['depth'] >= depth:
        val = tt_entry['value']
        if tt_entry['flag'] == 0: return val, tt_entry['move']
        elif tt_entry['flag'] == 1: alpha = max(alpha, val)
        elif tt_entry['flag'] == 2: beta = min(beta, val)
        if alpha >= beta: return val, tt_entry['move']
        tt_move = tt_entry['move']

    if depth <= 0: return evaluate(red_bb, blue_bb, is_red_turn), None

    moves = generate_moves(red_bb, blue_bb, is_red_turn)
    if not moves:
        nr, nb, nt, nh = make_move(red_bb, blue_bb, is_red_turn, hash_val, None)
        v, _ = negamax(nr, nb, nt, depth - 1, -beta, -alpha, nh, start_time, time_limit, ply + 1)
        return -v, None

    # MOVE ORDERING: TT -> Captures -> Killers -> Others
    k1, k2 = KILLERS[ply][0], KILLERS[ply][1]
    def m_score(m):
        if m == tt_move: return 100000
        if m[2]: return 10000
        if m == k1: return 5000
        if m == k2: return 4000
        return 0
    moves.sort(key=m_score, reverse=True)

    best_v, best_m, orig_alpha = -math.inf, None, alpha
    for move in moves:
        nr, nb, nt, nh = make_move(red_bb, blue_bb, is_red_turn, hash_val, move)
        val, _ = negamax(nr, nb, nt, depth - 1, -beta, -alpha, nh, start_time, time_limit, ply + 1)
        val = -val
        if val > best_v:
            best_v, best_m = val, move
        alpha = max(alpha, val)
        if alpha >= beta:
            # KILLER MOVE UPDATE
            if not move[2] and ply < 100:
                if KILLERS[ply][0] != move:
                    KILLERS[ply][1] = KILLERS[ply][0]
                    KILLERS[ply][0] = move
            break

    flag = 0 if best_v > orig_alpha and best_v < beta else (1 if best_v >= beta else 2)
    TT[hash_val] = {'depth': depth, 'value': best_v, 'move': best_m, 'flag': flag}
    return best_v, best_m

def playerStrategy(game, state, timeout=3):
    global TT, KILLERS, ACTIVE_WEIGHTS
    ACTIVE_WEIGHTS = BOT_WEIGHTS[0:34] if state.to_move == "Red" else BOT_WEIGHTS[34:68]
    if len(TT) > 1000000: TT.clear()
    for i in range(len(KILLERS)): KILLERS[i] = [None, None]
    
    start_time = time.time()
    time_limit = timeout - 0.15
    red_bb, blue_bb, is_red_turn = parse_state(state)
    h = get_hash(red_bb, blue_bb, is_red_turn)
    
    legal = generate_moves(red_bb, blue_bb, is_red_turn)
    if not legal: return None
    best_move, depth = random.choice(legal), 1
    
    while True:
        try:
            val, move = negamax(red_bb, blue_bb, is_red_turn, depth, -math.inf, math.inf, h, start_time, time_limit, 0)
            if move: best_move = move
            if val > 9000000 or val < -9000000: break
            depth += 1
        except TimeoutError: break
    
    fr, to, cap = best_move
    return (sq_to_rc(fr), sq_to_rc(to), cap)
