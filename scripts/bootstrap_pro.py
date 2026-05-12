import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(BASE_DIR, 'logs', 'best_pro_130.json')
state_path = os.path.join(BASE_DIR, 'logs', 'state_pro.json')

def bootstrap_aggressive():
    # 130 parametri:
    # 0..63: Rosso (valore caselle)
    # 64..127: Blu (valore caselle)
    # 128: Cluster Rosso
    # 129: Cluster Blu
    
    # Inizializziamo tutto a 1000 (Strategia Materialista Pura)
    pro_w = [1000.0] * 130
    
    # Piccola preferenza per il centro per rompere la simmetria (da 1 a 10 punti extra)
    for r in range(8):
        for c in range(8):
            dist_center = abs(3.5 - r) + abs(3.5 - c)
            bonus = (8 - dist_center) * 2
            pro_w[r*8 + c] += bonus
            pro_w[64 + r*8 + c] += bonus
            
    # Clustering (bonus difesa)
    pro_w[128] = 10.0
    pro_w[129] = 10.0
    
    # Salviamo sia come Best che come State per resettare il training
    data = {"gen": 0, "weights": pro_w}
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
        
    # Resettiamo anche la popolazione per il training futuro
    pop_data = {"gen": 0, "pop": [pro_w for _ in range(16)]} # Usiamo 16 come taglia standard
    with open(state_path, 'w') as f:
        json.dump(pop_data, f)
        
    print(f"BOOTSTRAP AGGRESSIVO COMPLETATO!")
    print(f"Il Pro130 ora parte con una strategia Materialista + Centro.")

if __name__ == "__main__":
    bootstrap_aggressive()
