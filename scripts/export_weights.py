import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    import torch
except ImportError:
    print("PyTorch non installato. Impossibile estrarre i pesi.")
    sys.exit(1)

import playerGPU
TARGET_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/player130params.py'))

def export():
    if not playerGPU.TORCH_AVAILABLE:
        print("PyTorch non disponibile in playerGPU!")
        return

    model = playerGPU.model
    
    try:
        # Pesi della rete neurale lineare: forma (1, 130)
        weights = model.fc.weight.data[0].cpu().numpy()
    except Exception as e:
        print(f"Errore nell'estrazione dei pesi: {e}")
        return
    
    # Primi 64 pesi per i pezzi Rossi, successivi 64 per i Blu
    red_weights = weights[:64].tolist()
    blue_weights = weights[64:128].tolist()
    
    # Ultimi 2 pesi per il Clustering
    peso_clustering_rosso = float(weights[128])
    peso_clustering_blu = float(weights[129])
    
    # 1. Salva in JSON per caricamento automatico
    json_file = os.path.join(os.path.dirname(__file__), '../logs/best_weights_130.json')
    import json
    with open(json_file, 'w') as f:
        json.dump({"weights": weights.tolist()}, f)
    
    # 2. Genera testo per copia-incolla manuale o debug
    out_file = os.path.join(os.path.dirname(__file__), '../logs/extracted_python_weights.txt')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    with open(out_file, 'w') as f:
        f.write("# ==================================================\n")
        f.write("# PESI GENERATI DALLA GPU (PyTorch Linear Model)\n")
        f.write("# Incolla questi in player130params.py se vuoi hardcoded\n")
        f.write("# ==================================================\n\n")
        f.write(f"PESI_ROSSO = {red_weights}\n")
        f.write(f"PESI_BLU = {blue_weights}\n")
        f.write(f"PESO_CLUSTERING_ROSSO = {peso_clustering_rosso:.3f}\n")
        f.write(f"PESO_CLUSTERING_BLU = {peso_clustering_blu:.3f}\n")
        
    print(f"Pesi estratti con successo!")
    print(f"- JSON salvato in: logs/best_weights_130.json (caricato automaticamente da player130params)")
    print(f"- Testo salvato in: logs/extracted_python_weights.txt (per hardcoding manuale)")
    print("Il tuo player130params.py è pronto per essere usato senza torch!")

if __name__ == "__main__":
    export()
