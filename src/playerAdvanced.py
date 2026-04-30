import time
import math
import random

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
random.seed(42) # Seed fisso per consistenza
ZOBRIST_RED = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_BLUE = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_TURN = random.getrandbits(64)

# Transposition Table globale
TT = {}

# Vettore di 68 Parametri:
# 0..33: Pesi per quando giochiamo per Primi (Rosso)
# 34..67: Pesi per quando giochiamo per Secondi (Blu)

DEFAULT_WEIGHTS = [0] * 68
DEFAULT_WEIGHTS[33] = 1000 # Material advantage First
DEFAULT_WEIGHTS[67] = 1000 # Material advantage Second

BOT_WEIGHTS = DEFAULT_WEIGHTS.copy()
ACTIVE_WEIGHTS = BOT_WEIGHTS[0:34] # Inizializzazione base

# Tenta di caricare i pesi ottimali generati dal training
try:
    import json
    import os
    weights_path = os.path.join(os.path.dirname(__file__), '../logs/best_weights.json')
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            data = json.load(f)
            best = data.get("best_weights")
            if best and len(best) == 68:
                BOT_WEIGHTS = best.copy()
except Exception:
    pass


# Opening Book Hardcoded per Trappole
# Essendo un gioco ad informazione perfetta, se giochiamo come Blue (secondi), possiamo
# pre-calcolare offline (es. lasciando il motore a profondità 20+ per una notte) le risposte
# alle prime mosse più probabili del Rosso, impostando delle vere e proprie "trappole".
OPENING_BOOK = {
    # Esempio di struttura:
    # hash_della_scacchiera_dopo_mossa_rosso: (fr_sq, to_sq, is_cap)
    # Se il rosso fa l'apertura A, noi rispondiamo istantaneamente con la contromossa B (la trappola).
}

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
        
        # 1. Mosse Non-Catturanti
        for target in ADJ_HIGHER[sq]:
            if not (occ_bb & BIT_MASKS[target]):
                moves.append((sq, target, False))
                
        # 2. Mosse Catturanti
        sq_lvl = LEVELS[sq]
        for d_idx in range(8):
            for target in RAYS[sq][d_idx]:
                mask = BIT_MASKS[target]
                if occ_bb & mask:
                    # Incontrato un pezzo. Se nemico e livello <=, è cattura.
                    if opp_bb & mask:
                        if LEVELS[target] <= sq_lvl:
                            moves.append((sq, target, True))
                    break # Raggio bloccato, non andiamo oltre
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
    total_pieces = red_count + blue_count
    if total_pieces > 48:
        phase_offset = 0 # Early
    elif total_pieces > 24:
        phase_offset = 10 # Mid
    else:
        phase_offset = 20 # Late
        
    score = (red_count * ACTIVE_WEIGHTS[33]) - (blue_count * ACTIVE_WEIGHTS[33])
    
    # Valutazione Posizionale (Fasi di Gioco)
    r_bb = red_bb
    while r_bb:
        lsb = r_bb & -r_bb
        sq = lsb.bit_length() - 1
        r_bb ^= lsb
        score += ACTIVE_WEIGHTS[phase_offset + LEVELS[sq]]
        
    b_bb = blue_bb
    while b_bb:
        lsb = b_bb & -b_bb
        sq = lsb.bit_length() - 1
        b_bb ^= lsb
        score -= ACTIVE_WEIGHTS[phase_offset + LEVELS[sq]]
        
    # Parametro 31: Clustering (Pezzi adiacenti che si difendono)
    red_cluster = (red_bb & (red_bb << 1)).bit_count() + (red_bb & (red_bb >> 1)).bit_count() + \
                  (red_bb & (red_bb << 8)).bit_count() + (red_bb & (red_bb >> 8)).bit_count()
    blue_cluster = (blue_bb & (blue_bb << 1)).bit_count() + (blue_bb & (blue_bb >> 1)).bit_count() + \
                   (blue_bb & (blue_bb << 8)).bit_count() + (blue_bb & (blue_bb >> 8)).bit_count()
                   
    score += red_cluster * ACTIVE_WEIGHTS[31]
    score -= blue_cluster * ACTIVE_WEIGHTS[31]
    
    return score if is_red_turn else -score

# ==========================================
# MOTORE DI RICERCA (PVS + ITERATIVE DEEPENING)
# ==========================================
def negamax(red_bb, blue_bb, is_red_turn, depth, alpha, beta, hash_val, start_time, time_limit):
    if time.time() - start_time > time_limit:
        raise TimeoutError()
        
    # Condizioni di vittoria immediate
    if red_bb == 0:
        base_score = -1000000 + (100 - depth)
        return (base_score if is_red_turn else -base_score), None
    if blue_bb == 0:
        base_score = 1000000 - (100 - depth)
        return (base_score if is_red_turn else -base_score), None
        
    # Transposition Table Probe
    tt_entry = TT.get(hash_val)
    if tt_entry and tt_entry['depth'] >= depth:
        flag = tt_entry['flag']
        value = tt_entry['value']
        if flag == 0: # EXACT
            return value, tt_entry['best_move']
        elif flag == 1: # LOWERBOUND
            alpha = max(alpha, value)
        elif flag == 2: # UPPERBOUND
            beta = min(beta, value)
        if alpha >= beta:
            return value, tt_entry['best_move']
            
    if depth == 0:
        # TODO: Implementare Quiescence Search qui se il tempo lo permette
        return evaluate(red_bb, blue_bb, is_red_turn), None
        
    moves = generate_moves(red_bb, blue_bb, is_red_turn)
    if not moves:
        n_red, n_blue, n_turn, n_hash = make_move(red_bb, blue_bb, is_red_turn, hash_val, None)
        val, _ = negamax(n_red, n_blue, n_turn, depth - 1, -beta, -alpha, n_hash, start_time, time_limit)
        return -val, None
        
    # Move Ordering (Tagli Alpha-Beta più veloci)
    tt_best = tt_entry['best_move'] if tt_entry else None
    
    def move_score(m):
        if m == tt_best: return 10000 # TT move per prima
        if m[2]: return 1000 # Catture per seconde
        return 0 # Mosse silenziose
        
    moves.sort(key=move_score, reverse=True)
    
    best_move = None
    orig_alpha = alpha
    best_val = -math.inf
    
    for i, move in enumerate(moves):
        n_red, n_blue, n_turn, n_hash = make_move(red_bb, blue_bb, is_red_turn, hash_val, move)
        
        # Principal Variation Search (PVS)
        if i == 0:
            val, _ = negamax(n_red, n_blue, n_turn, depth - 1, -beta, -alpha, n_hash, start_time, time_limit)
            val = -val
        else:
            # Ricerca a Finestra Nulla
            val, _ = negamax(n_red, n_blue, n_turn, depth - 1, -alpha - 1, -alpha, n_hash, start_time, time_limit)
            val = -val
            if alpha < val < beta:
                # Ricerca Completa se fallisce
                val, _ = negamax(n_red, n_blue, n_turn, depth - 1, -beta, -val, n_hash, start_time, time_limit)
                val = -val
                
        if val > best_val:
            best_val = val
            best_move = move
            
        alpha = max(alpha, val)
        if alpha >= beta:
            break # Cutoff
            
    # Salva nella Transposition Table
    flag = 0 # EXACT
    if best_val <= orig_alpha:
        flag = 2 # UPPERBOUND
    elif best_val >= beta:
        flag = 1 # LOWERBOUND
        
    TT[hash_val] = {
        'depth': depth,
        'flag': flag,
        'value': best_val,
        'best_move': best_move
    }
    
    return best_val, best_move

# ==========================================
# INTERFACCIA PRINCIPALE
# ==========================================
def playerStrategy(game, state, timeout=3):
    global TT, ACTIVE_WEIGHTS
    # Seleziona il cervello (Primo o Secondo giocatore) in base a chi siamo
    if state.to_move == "Red":
        ACTIVE_WEIGHTS = BOT_WEIGHTS[0:34]
    else:
        ACTIVE_WEIGHTS = BOT_WEIGHTS[34:68]
        
    # Salvaguardia RAM (32GB / multi-core): Pulisce la cache se diventa critica
    if len(TT) > 5_000_000:
        TT.clear()
        
    start_time = time.time()
    time_limit = timeout - 0.15 # Limite di sicurezza (2.85 secondi)
    
    red_bb, blue_bb, is_red_turn = parse_state(state)
    hash_val = get_initial_hash(red_bb, blue_bb, is_red_turn)
    
    # 1. Controlla l'Opening Book (Trappole)
    if hash_val in OPENING_BOOK:
        fr_sq, to_sq, is_cap = OPENING_BOOK[hash_val]
        return (sq_to_rc(fr_sq), sq_to_rc(to_sq), is_cap)
    
    # 2. Controllo Mosse Legali
    legal_moves = generate_moves(red_bb, blue_bb, is_red_turn)
    if not legal_moves:
        return None # Il game engine gestirà il turno saltato
        
    best_move = None
    depth = 1
    
    # 3. Iterative Deepening
    while True:
        try:
            val, move = negamax(red_bb, blue_bb, is_red_turn, depth, -math.inf, math.inf, hash_val, start_time, time_limit)
            if move is not None:
                best_move = move
            
            # L'AI sfrutterà tutti i 2.8 secondi, rimuoviamo l'interruzione anticipata
            # in modo che continui a valutare la mossa in maggiore profondità
            # if val > 900000 or val < -900000:
            #     break
                
            depth += 1
        except TimeoutError:
            # print(f"[Zola AI] Timeout raggiunto alla profondità {depth}")
            break
            
    # Fallback estremo
    if best_move is None:
        best_move = random.choice(legal_moves)
        
    fr_sq, to_sq, is_cap = best_move
    return (sq_to_rc(fr_sq), sq_to_rc(to_sq), is_cap)
