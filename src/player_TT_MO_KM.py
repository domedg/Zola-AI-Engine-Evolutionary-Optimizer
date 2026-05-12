import time
import random


class TimeOutException(Exception):
    pass


# =====================================================================
# 1. MEMORIA GLOBALE (TT e Killer Moves)
# =====================================================================
TT = {}
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2

# Salviamo 2 Killer Moves per ogni livello di profondità (fino a 100 livelli, ampiamente sufficiente)
KILLERS = [[None, None] for _ in range(100)]


def get_state_hash(state):
    return hash((tuple(tuple(row) for row in state.board), state.to_move))


def evaluate_board(game, state, player, depth):
    winner = game.winner(state)
    if winner is not None:
        return (10000000 + depth) if winner == player else (-10000000 - depth)

    enemy = "Blue" if player == "Red" else "Red"

    my_pieces = 0
    enemy_pieces = 0
    my_high_ground = 0
    enemy_high_ground = 0

    for r in range(state.size):
        for c in range(state.size):
            cell = state.board[r][c]
            if cell is not None:
                level = game.get_distance_level(r, c)
                if cell == player:
                    my_pieces += 1
                    my_high_ground += level
                else:
                    enemy_pieces += 1
                    enemy_high_ground += level

    material_score = my_pieces - enemy_pieces
    position_score = my_high_ground - enemy_high_ground

    return (material_score * 100000) + (position_score * 10)


def alpha_beta_search(game, state, depth, alpha, beta, start_time, time_limit, root_player, is_maximizing):
    if (time.perf_counter() - start_time) > time_limit:
        raise TimeOutException()

    state_hash = get_state_hash(state)
    tt_entry = TT.get(state_hash)
    tt_best_move = None

    if tt_entry:
        tt_depth, tt_flag, tt_val, tt_best_move = tt_entry
        if tt_depth >= depth:
            if tt_flag == EXACT:
                return tt_val, tt_best_move
            elif tt_flag == LOWERBOUND:
                alpha = max(alpha, tt_val)
            elif tt_flag == UPPERBOUND:
                beta = min(beta, tt_val)
            if alpha >= beta: return tt_val, tt_best_move

    if depth == 0 or game.is_terminal(state):
        return evaluate_board(game, state, root_player, depth), None

    legal_moves = game.actions(state)

    if not legal_moves:
        next_state = game.pass_turn(state)
        val, _ = alpha_beta_search(game, next_state, depth - 1, alpha, beta, start_time, time_limit, root_player,
                                   not is_maximizing)
        return val, "PASS"

    # =====================================================================
    # 2. MOVE ORDERING DEFINITIVO (TT -> Catture -> Killer -> Altre)
    # =====================================================================
    random.shuffle(legal_moves)
    ordered_moves = []

    if tt_best_move and tt_best_move in legal_moves:
        ordered_moves.append(tt_best_move)
        legal_moves.remove(tt_best_move)

    captures = []
    killers = []
    others = []

    # Recuperiamo le killer moves per la profondità corrente
    killer1, killer2 = KILLERS[depth][0], KILLERS[depth][1]

    for m in legal_moves:
        if m[2]:  # È una cattura
            captures.append(m)
        elif m == killer1 or m == killer2:  # È una mossa Killer
            killers.append(m)
        else:  # Mossa normale
            others.append(m)

    ordered_moves.extend(captures)
    ordered_moves.extend(killers)
    ordered_moves.extend(others)
    # =====================================================================

    best_move = ordered_moves[0]
    orig_alpha = alpha
    orig_beta = beta

    if is_maximizing:
        max_eval = -float('inf')
        for move in ordered_moves:
            next_state = game.result(state, move)
            eval_score, _ = alpha_beta_search(game, next_state, depth - 1, alpha, beta, start_time, time_limit,
                                              root_player, False)

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)

            # --- TAGLIO ALPHA-BETA E AGGIORNAMENTO KILLER MOVES ---
            if beta <= alpha:
                if not move[2]:  # Le catture non sono killer moves
                    if KILLERS[depth][0] != move:
                        KILLERS[depth][1] = KILLERS[depth][0]
                        KILLERS[depth][0] = move
                break

        if max_eval <= orig_alpha:
            flag = UPPERBOUND
        elif max_eval >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        TT[state_hash] = (depth, flag, max_eval, best_move)

        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            next_state = game.result(state, move)
            eval_score, _ = alpha_beta_search(game, next_state, depth - 1, alpha, beta, start_time, time_limit,
                                              root_player, True)

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)

            # --- TAGLIO ALPHA-BETA E AGGIORNAMENTO KILLER MOVES ---
            if beta <= alpha:
                if not move[2]:  # Le catture non sono killer moves
                    if KILLERS[depth][0] != move:
                        KILLERS[depth][1] = KILLERS[depth][0]
                        KILLERS[depth][0] = move
                break

        if min_eval <= alpha:
            flag = UPPERBOUND
        elif min_eval >= orig_beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        TT[state_hash] = (depth, flag, min_eval, best_move)

        return min_eval, best_move


def playerStrategy(game, state, time_out):
    start_time = time.perf_counter()
    TIME_LIMIT = time_out - 0.2

    # Pulizia memorie ad ogni turno
    TT.clear()
    for i in range(len(KILLERS)):
        KILLERS[i] = [None, None]

    legal_moves = game.actions(state)
    if not legal_moves:
        return "PASS"

    best_move_overall = random.choice(legal_moves)
    root_player = state.to_move

    depth = 1
    score = 0

    try:
        while True:
            if (time.perf_counter() - start_time) > TIME_LIMIT:
                break

            current_score, best_move = alpha_beta_search(
                game, state, depth, -float('inf'), float('inf'),
                start_time, TIME_LIMIT, root_player, True
            )

            if best_move is not None:
                best_move_overall = best_move
                score = current_score

            if score >= 10000000 or score <= -10000000:
                break

            depth += 1

    except TimeOutException:
        pass

    print(f"[{root_player} BOSS+TT+ORDER+KM] Prof: {depth - 1} | Score: {score}")

    return best_move_overall