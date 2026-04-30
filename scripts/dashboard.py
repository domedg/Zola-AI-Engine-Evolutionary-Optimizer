import streamlit as st
import pandas as pd
import json
import os
import time
import numpy as np

st.set_page_config(page_title="Zola AI Optimizer", page_icon="🧬", layout="wide")

st.title("🧬 Zola AI: Advanced Evolutionary Dashboard")
st.markdown("Monitoraggio approfondito dell'addestramento. Scopri **come pensa** l'intelligenza artificiale.")

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, '../logs/training_history.csv')
json_path = os.path.join(base_dir, '../logs/training_state.json')

st.sidebar.header("📊 Stato del Sistema")
if os.path.exists(json_path):
    try:
        with open(json_path, 'r') as f:
            state = json.load(f)
            gen = state.get("generation", 0)
            st.sidebar.metric(label="Generazione Attuale", value=gen)
    except Exception:
        pass

st.sidebar.button("Aggiorna Dati", on_click=lambda: time.sleep(0.1))

# Funzioni helper per estrarre le medie posizionali
def get_avg(pesi, start, end):
    subset = pesi[start:end+1]
    return sum(subset) / len(subset) if subset else 0

history_data = []
if os.path.exists(json_path):
    try:
        with open(json_path, 'r') as f:
            state = json.load(f)
            archive = state.get("archive", [])
            for idx, pesi in enumerate(archive):
                if len(pesi) >= 68:
                    history_data.append({
                        "Generation": idx + 1,
                        "Score": None,
                        "Material_First": pesi[33],
                        "Material_Second": pesi[67],
                        "Clustering_First": pesi[31],
                        "Clustering_Second": pesi[65],
                        # Medie posizionali First
                        "Early_First": get_avg(pesi, 1, 9),
                        "Mid_First": get_avg(pesi, 11, 19),
                        "Late_First": get_avg(pesi, 21, 29),
                        # Medie posizionali Second
                        "Early_Second": get_avg(pesi, 35, 43),
                        "Mid_Second": get_avg(pesi, 45, 53),
                        "Late_Second": get_avg(pesi, 55, 63)
                    })
    except Exception:
        pass

csv_data = []
if os.path.exists(csv_path):
    try:
        df_csv = pd.read_csv(csv_path)
        csv_data = df_csv.to_dict('records')
    except Exception:
        pass

merged_dict = {}
for item in history_data: merged_dict[item["Generation"]] = item
for item in csv_data:
    gen = item["Generation"]
    if gen in merged_dict:
        merged_dict[gen]["Score"] = item["Score"]
        merged_dict[gen]["Material_First"] = item.get("Material_First", merged_dict[gen]["Material_First"])
        merged_dict[gen]["Material_Second"] = item.get("Material_Second", merged_dict[gen]["Material_Second"])
        merged_dict[gen]["Clustering_First"] = item.get("Clustering_First", merged_dict[gen]["Clustering_First"])
        merged_dict[gen]["Clustering_Second"] = item.get("Clustering_Second", merged_dict[gen]["Clustering_Second"])
    else:
        # Se nel CSV non ci sono i dati posizionali dettagliati (non li salvavamo), li mettiamo a 0 o li saltiamo
        merged_dict[gen] = item

if not merged_dict:
    st.info("Attendi la fine della prima generazione.")
else:
    df = pd.DataFrame(list(merged_dict.values())).sort_values(by="Generation")
    
    st.markdown("---")
    st.header("📈 1. Curva di Dominanza (Punteggio Complessivo)")
    st.info("**Cosa stai guardando:** L'area rossa rappresenta il Punteggio Totale del bot Campione in ogni generazione.\n\n"
            "**Cosa significa:** Il punteggio viene calcolato facendo giocare il bot contro tutti i suoi 'fratelli' mutati e contro un'Intelligenza Artificiale esterna (l'Alpha Baseline). Un punteggio più alto significa che l'IA ha scoperto una mutazione che le permette di stravincere contro le strategie vecchie.\n\n"
            "**Come interpretarlo:** Quando l'area smette di crescere e diventa una linea retta per molte generazioni (il *Plateau*), significa che l'IA ha raggiunto la perfezione matematica rispetto alle sue possibilità e non riesce a trovare mutazioni migliori. È il momento in cui l'addestramento è concluso.")
    df_score = df.dropna(subset=["Score"])
    if not df_score.empty:
        st.area_chart(df_score.set_index("Generation")["Score"], color="#FF4B4B")
    
    st.markdown("---")
    st.header("🔍 2. Profilo Tattico Istantaneo (Ultima Mutazione)")
    st.info("**Cosa stai guardando:** I valori esatti del Campione in carica (l'ultima generazione calcolata). I numerini verdi/rossi sotto ogni valore indicano la 'Mutazione', ovvero quanto quel valore è cambiato rispetto al padre (la generazione precedente).\n\n"
            "**Come interpretarlo:** Se vedi '+15' verde sul Materiale (Secondo), significa che l'ultimo bot ha capito che giocando per secondo deve essere più aggressivo e mangiare più pedine rispetto a prima.")
    
    latest = df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Fame di Materiale (Primo)", int(latest["Material_First"]), int(latest["Material_First"] - df.iloc[-2]["Material_First"]) if len(df)>1 else 0)
    col1.metric("Fame di Materiale (Secondo)", int(latest["Material_Second"]), int(latest["Material_Second"] - df.iloc[-2]["Material_Second"]) if len(df)>1 else 0)
    
    col2.metric("Istinto di Branco (Clustering Primo)", int(latest["Clustering_First"]), int(latest["Clustering_First"] - df.iloc[-2]["Clustering_First"]) if len(df)>1 else 0)
    col2.metric("Istinto di Branco (Clustering Secondo)", int(latest["Clustering_Second"]), int(latest["Clustering_Second"] - df.iloc[-2]["Clustering_Second"]) if len(df)>1 else 0)
    
    st.markdown("---")
    st.header("🌍 3. Evoluzione della Mentalità (Fasi della Partita)")
    st.info("**Cosa stai guardando:** Come cambia l'aggressività del bot in base al momento della partita: Apertura (Early), Metà Gioco (Mid) e Finale (Late).\n\n"
            "**Cosa significa:** Il motore AI sa contare quante pedine sono in campo. Più le linee sono alte nel grafico, più il bot dà importanza alle 'posizioni' strategiche rispetto al mangiare le pedine.\n\n"
            "**Come interpretarlo:** Un bot maturo di solito ha una curva 'Early' molto alta (vuole posizionarsi bene all'inizio) e una curva 'Late' più bassa (nel finale non importa la posizione, importa solo sbranare l'avversario).")
    
    tab1, tab2 = st.tabs(["Primo Giocatore (Mentalità d'Attacco)", "Secondo Giocatore (Mentalità di Difesa)"])
    with tab1:
        st.line_chart(df.set_index("Generation")[["Early_First", "Mid_First", "Late_First"]])
    with tab2:
        st.line_chart(df.set_index("Generation")[["Early_Second", "Mid_Second", "Late_Second"]])
        
    st.markdown("---")
    st.header("🔬 4. Radiografia Posizionale (L'impronta Digitale del Bot)")
    st.info("**Cosa stai guardando:** Questo è il nuovo grafico a Barre! Mostra esattamente quanto il bot valuta le 9 'zone' della scacchiera nelle diverse fasi della partita (Early, Mid, Late).\n\n"
            "**Cosa significa:** Ogni colonna rappresenta il peso (positivo o negativo) che il bot assegna a una determinata posizione.\n\n"
            "**Come interpretarlo:** Guardando queste barre puoi capire esattamente cosa pensa il bot. Se la barra 'Early' è molto alta, ma la 'Late' è negativa nella stessa zona, significa che l'IA ha capito che quella casella è fantastica a inizio partita ma diventa una trappola mortale nel finale!")
    
    if history_data:
        # Prende i pesi originali dell'ultimo campione dall'archivio
        last_weights = []
        try:
            with open(json_path, 'r') as f:
                state = json.load(f)
                archive = state.get("archive", [])
                if archive:
                    last_weights = archive[-1]
        except Exception:
            pass
            
        if len(last_weights) >= 68:
            # Creiamo un DataFrame per le 9 zone posizionali (Primo Giocatore)
            zones = [f"Zona {i}" for i in range(1, 10)]
            bar_df = pd.DataFrame({
                "Apertura (Early)": last_weights[1:10],
                "Metà Partita (Mid)": last_weights[11:20],
                "Finale (Late)": last_weights[21:30]
            }, index=zones)
            
            st.bar_chart(bar_df)
    
    st.markdown("---")
    st.header("⚔️ 5. Focus: Materiale vs Formazione")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Materiale:** Se la linea sale, il bot sta diventando più ingordo.")
        st.line_chart(df.set_index("Generation")[["Material_First", "Material_Second"]])
    with col4:
        st.markdown("**Clustering:** Se la linea sale, il bot preferisce tenere i pezzi uniti a falange o testuggine.")
        st.line_chart(df.set_index("Generation")[["Clustering_First", "Clustering_Second"]])
