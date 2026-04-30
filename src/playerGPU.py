import time
import math
import random
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from playerAdvanced import parse_state, generate_moves, make_move, sq_to_rc, BIT_MASKS

# ==========================================
# NEURAL NETWORK EVALUATOR
# ==========================================
if TORCH_AVAILABLE:
    class ZolaEvaluator(nn.Module):
        def __init__(self):
            super().__init__()
            # A completely linear model matching playerAdvanced.py mathematical structure.
            # 64 positional weights for Red, 64 positional weights for Blue, 1 Red Cluster, 1 Blue Cluster = 130 parameters
            self.fc = nn.Linear(130, 1, bias=False)
            
        def forward(self, x):
            return self.fc(x).squeeze(1)

    # Setup Device (NVIDIA CUDA, AMD ROCm, Apple MPS, or CPU fallback)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = ZolaEvaluator().to(device)
    model.eval()

    weights_path = os.path.join(os.path.dirname(__file__), '../logs/gpu_weights.pth')
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"[Zola GPU] Loaded pre-trained weights from {weights_path}")
        except Exception as e:
            print(f"[Zola GPU] Architettura modificata. Elimino vecchi pesi incompatibili: {e}")
            os.remove(weights_path)
    else:
        print("[Zola GPU] No pre-trained weights found. Using random initialization.")

    # Precompute shifts for rapid bitboard-to-tensor conversion
    SHIFTS = torch.arange(64, device=device)

    def bitboards_to_tensor(red_bbs, blue_bbs):
        N = len(red_bbs)
        if N == 0:
            return torch.empty((0, 2, 8, 8), device=device)
            
        red_np = np.array(red_bbs, dtype=np.uint64).view(np.int64)
        blue_np = np.array(blue_bbs, dtype=np.uint64).view(np.int64)
        
        red_t = torch.tensor(red_np, device=device).unsqueeze(1)
        blue_t = torch.tensor(blue_np, device=device).unsqueeze(1)
        
        red_bits = ((red_t >> SHIFTS) & 1).float().squeeze(1)
        blue_bits = ((blue_t >> SHIFTS) & 1).float().squeeze(1)
        
        # Calculate Clustering mathematically matching playerAdvanced.py
        # playerAdvanced: (bb & (bb << 1)) + (bb & (bb >> 1)) + (bb & (bb << 8)) + (bb & (bb >> 8))
        r_c1 = (red_bits[:, :-1] * red_bits[:, 1:]).sum(dim=1, keepdim=True)
        r_c8 = (red_bits[:, :-8] * red_bits[:, 8:]).sum(dim=1, keepdim=True)
        red_cluster = r_c1 * 2 + r_c8 * 2
        
        b_c1 = (blue_bits[:, :-1] * blue_bits[:, 1:]).sum(dim=1, keepdim=True)
        b_c8 = (blue_bits[:, :-8] * blue_bits[:, 8:]).sum(dim=1, keepdim=True)
        blue_cluster = b_c1 * 2 + b_c8 * 2
        
        # Concatenate into (N, 130) tensor
        tensor = torch.cat([red_bits, blue_bits, red_cluster, blue_cluster], dim=1)
        return tensor

    def batched_evaluate(red_bbs, blue_bbs):
        with torch.no_grad():
            x = bitboards_to_tensor(red_bbs, blue_bbs)
            scores = model(x)
            return scores.cpu().numpy()

# ==========================================
# BATCHED MINIMAX SEARCH
# ==========================================

GLOBAL_TT = {}

def build_tree(red_bb, blue_bb, is_red_turn, depth, node_counter, start_time, time_limit, visited):
    state_key = (red_bb, blue_bb, is_red_turn, depth)
    if state_key in visited:
        return visited[state_key]
        
    node = {'red_bb': red_bb, 'blue_bb': blue_bb, 'turn': is_red_turn, 'children': []}
    visited[state_key] = node
    
    if time.time() - start_time > time_limit:
        node['timeout'] = True
        return node
        
    global GLOBAL_TT
    board_key = (red_bb, blue_bb, is_red_turn)
    if board_key in GLOBAL_TT:
        stored_depth, stored_val = GLOBAL_TT[board_key]
        if stored_depth >= depth:
            node['value'] = stored_val
            return node # Exact value known from previous deep search!
            
    if depth == 0 or red_bb == 0 or blue_bb == 0:
        return node
        
    moves = generate_moves(red_bb, blue_bb, is_red_turn)
    
    if not moves:
        n_red, n_blue, n_turn, _ = make_move(red_bb, blue_bb, is_red_turn, 0, None)
        node_counter[0] += 1
        node['children'].append((None, build_tree(n_red, n_blue, n_turn, depth - 1, node_counter, start_time, time_limit, visited)))
    else:
        for m in moves:
            n_red, n_blue, n_turn, _ = make_move(red_bb, blue_bb, is_red_turn, 0, m)
            node_counter[0] += 1
            node['children'].append((m, build_tree(n_red, n_blue, n_turn, depth - 1, node_counter, start_time, time_limit, visited)))
            
    return node

def propagate_values(node):
    if 'value' in node:
        return node['value']
        
    if not node['children']:
        return node.get('value', 0)
        
    # Red maximizes, Blue minimizes
    best = -math.inf if node['turn'] else math.inf
    
    for move, child in node['children']:
        val = propagate_values(child)
        if node['turn']: 
            if val > best: best = val
        else:
            if val < best: best = val
            
    node['value'] = best
    return best

def gpu_search(game, state, timeout=3.0):
    global GLOBAL_TT
    
    start_time = time.time()
    time_limit = timeout - 0.2 # Safety buffer
    
    if len(GLOBAL_TT) > 1_500_000:
        GLOBAL_TT.clear() # Prevent Memory Leak
        
    red_bb, blue_bb, is_red_turn = parse_state(state)
    
    legal_moves = generate_moves(red_bb, blue_bb, is_red_turn)
    if not legal_moves:
        return None
        
    best_move = random.choice(legal_moves)
    depth = 1
    
    while True:
        # Check if we have barely any time left before starting a new depth
        if time.time() - start_time > time_limit:
            break
            
        node_counter = [1]
        visited = {}
        # We give build_tree slightly less time to ensure we have time to evaluate
        tree = build_tree(red_bb, blue_bb, is_red_turn, depth, node_counter, start_time, time_limit - 0.5, visited)
        
        # Extract unique leaves
        leaves = [node for node in visited.values() if not node['children']]
        
        # If we timed out during build, abandon this depth (unless it's depth 1)
        if depth > 1 and (leaves and leaves[0].get('timeout')):
            break
            
        # Batch Evaluation
        # If the professor runs this on a CPU without CUDA, 50k batch would take too long.
        # We shrink the chunk size on CPU so the time check happens much more frequently!
        chunk_size = 50000 if device.type == 'cuda' else 2000
        timeout_during_eval = False
        
        for i in range(0, len(leaves), chunk_size):
            if time.time() - start_time > time_limit:
                timeout_during_eval = True
                break
                
            chunk = leaves[i:i+chunk_size]
            r_bbs = [leaf['red_bb'] for leaf in chunk]
            b_bbs = [leaf['blue_bb'] for leaf in chunk]
            
            scores = batched_evaluate(r_bbs, b_bbs)
            
            for leaf, score in zip(chunk, scores):
                if leaf['blue_bb'] == 0: 
                    leaf['value'] = 1000000
                elif leaf['red_bb'] == 0: 
                    leaf['value'] = -1000000
                else: 
                    leaf['value'] = float(score)
        
        if timeout_during_eval and depth > 1:
            break # We didn't finish evaluating this depth, discard it
                    
        # Propagate Minimax Values
        best_val = -math.inf if is_red_turn else math.inf
        current_best_move = None
        
        for move, child in tree['children']:
            val = propagate_values(child)
            if is_red_turn:
                if val > best_val:
                    best_val = val
                    current_best_move = move
            else:
                if val < best_val:
                    best_val = val
                    current_best_move = move
                    
        # Store exact values in persistent TT for future searches
        for key, node in visited.items():
            if 'value' in node and not node.get('timeout'):
                b_key = (node['red_bb'], node['blue_bb'], node['turn'])
                n_depth = key[3]
                
                # Only update if we searched deeper or didn't have it
                if b_key not in GLOBAL_TT or GLOBAL_TT[b_key][0] <= n_depth:
                    GLOBAL_TT[b_key] = (n_depth, node['value'])
                    
        if current_best_move is not None:
            best_move = current_best_move
            
        # If we found a forced win/loss, we can break early
        if best_val >= 900000 or best_val <= -900000:
            break
            
        depth += 1
        
    fr_sq, to_sq, is_cap = best_move
    return (sq_to_rc(fr_sq), sq_to_rc(to_sq), is_cap)

# ==========================================
# INTERFACE
# ==========================================
def playerStrategy(game, state, timeout=3):
    if not TORCH_AVAILABLE:
        print("[Zola GPU] PyTorch non disponibile. Eseguo fallback su playerAdvanced!")
        import playerAdvanced as cpu_bot
        return cpu_bot.playerStrategy(game, state, timeout)
        
    return gpu_search(game, state, timeout)
