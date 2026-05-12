import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
state_path = os.path.join(BASE_DIR, 'logs', 'state_pro.json')
best_path = os.path.join(BASE_DIR, 'logs', 'best_pro_130.json')

def devalue():
    if not os.path.exists(state_path):
        print("Errore: state_pro.json non trovato.")
        return

    with open(state_path, 'r') as f:
        data = json.load(f)
    
    # Processiamo la popolazione
    new_pop = []
    for weights in data['pop']:
        new_w = list(weights)
        for i in range(128): # Le prime 128 sono le caselle
            # Portiamo la base da 1000 a 100, mantenendo la variazione (mutazione)
            # Esempio: 1015 diventa 115
            new_w[i] = (new_w[i] - 900)
            if new_w[i] < 10: new_w[i] = 10 # Minimo sindacale per un pezzo
        new_pop.append(new_w)
    
    data['pop'] = new_pop
    with open(state_path, 'w') as f:
        json.dump(data, f)
        
    # Facciamo lo stesso per il file del "best"
    if os.path.exists(best_path):
        with open(best_path, 'r') as f:
            best_data = json.load(f)
        best_w = list(best_data['weights'])
        for i in range(128):
            best_w[i] = (best_w[i] - 900)
            if best_w[i] < 10: best_w[i] = 10
        best_data['weights'] = best_w
        with open(best_path, 'w') as f:
            json.dump(best_data, f)

    print("SVALUTAZIONE COMPLETATA!")
    print("Ora le pedine valgono 100 punti invece di 1000. La strategia è 10 volte più importante!")

if __name__ == "__main__":
    devalue()
