"""Gemello digitale del riscaldamento ohmico multifase: punto di ingresso dell'app (navigazione tra le pagine).

Avvio locale:   streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Gemello digitale ohmico", page_icon=":material/bolt:", layout="wide", initial_sidebar_state="expanded")
from core import ui
ui.applica_tema()

pagine = st.navigation({
    "Panoramica": [st.Page("pagine/panoramica.py", title="Panoramica", icon=":material/dashboard:", url_path="panoramica", default=True)],
    "Cella e modello": [st.Page("pagine/cella_dati.py", title="Cella, ricette e dati", icon=":material/science:", url_path="cella-dati"),
                        st.Page("pagine/simulatore.py", title="Simulatore", icon=":material/play_circle:", url_path="simulatore"),
                        st.Page("pagine/nominale.py", title="Modello nominale e misure", icon=":material/compare_arrows:", url_path="nominale")],
    "Studi numerici": [st.Page("pagine/verifica.py", title="Verifica numerica e incertezza", icon=":material/grid_on:", url_path="verifica"),
                       st.Page("pagine/sensitivita.py", title="Sensitività", icon=":material/tune:", url_path="sensitivita"),
                       st.Page("pagine/dossier.py", title="Dossier del caso", icon=":material/folder_open:", url_path="dossier")],
    "Calibrazione e validazione": [st.Page("pagine/calibrazione.py", title="Calibrazione", icon=":material/target:", url_path="calibrazione"),
                                   st.Page("pagine/validazione.py", title="Validazione", icon=":material/verified:", url_path="validazione")],
    "Applicazioni": [st.Page("pagine/scenari.py", title="Scenari e incertezza", icon=":material/insights:", url_path="scenari")],
    "Strumenti": [st.Page("pagine/modello_base.py", title="Modello base monofase", icon=":material/deployed_code:", url_path="modello-base"),
                  st.Page("pagine/guida.py", title="Guida e metodi", icon=":material/menu_book:", url_path="guida")],
})
with st.sidebar:
    st.markdown("### ⚡ Gemello digitale ohmico")
    st.caption("Cella 99 × 54 × 66 mm · purea e polpette · 5 ricette, 5 configurazioni, 30–50 V")
pagine.run()
