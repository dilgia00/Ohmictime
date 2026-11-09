"""Caricamento dei dati esportati (risultati del protocollo e dati sperimentali), con cache."""
import os
import numpy as np
import pandas as pd
import streamlit as st

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(AQUI, 'data')
AB = {'A': (0.0643, 1.9794), 'B': (0.1091, 3.6907), 'C': (0.0067, 0.2275), 'D': (0.0247, 0.3037), 'E': (0.0390, 0.5290)}        # sigma(T) = a T + b della purea (S/m)
SALE = {'A': 2.9, 'B': 5.7, 'C': 0.0, 'D': 0.725, 'E': 1.45}
SIGMA_POLP = (0.0340, 0.6087)
CONTR = {r: (SIGMA_POLP[0] * 20 + SIGMA_POLP[1]) / (a * 20 + b) for r, (a, b) in AB.items()}
RICS = sorted(CONTR, key=lambda r: CONTR[r])                      # ordine di contrasto crescente: B, A, E, D, C
CFG = ['Tris', 'Symm', 'Asymm', 'Middle', 'Side']; VS = (30, 40, 50)
NPOLP = {'Tris': 3, 'Symm': 2, 'Asymm': 2, 'Middle': 1, 'Side': 1}
OCC = {'Tris': (1, 1, 1), 'Symm': (1, 0, 1), 'Asymm': (1, 1, 0), 'Middle': (0, 1, 0), 'Side': (1, 0, 0)}
PAR = ['sig_m20', 'alfa_m', 'sig_p20', 'alfa_p', 'fk_m', 'fk_p', 'fc_m', 'fc_p', 'fh']
PAR_TXT = {'sig_m20': 'σ₂₀,m', 'alfa_m': 'α_m', 'sig_p20': 'σ₂₀,p', 'alfa_p': 'α_p', 'fk_m': 'f_k,m', 'fk_p': 'f_k,p', 'fc_m': 'f_c,m', 'fc_p': 'f_c,p', 'fh': 'f_h'}
PAR_DESC = {'sig_m20': 'Conducibilità elettrica della purea a 20 °C', 'alfa_m': 'Coefficiente di temperatura della conducibilità della purea', 'sig_p20': 'Conducibilità elettrica delle polpette a 20 °C', 'alfa_p': 'Coefficiente di temperatura della conducibilità delle polpette',
            'fk_m': 'Fattore sulla conducibilità termica della purea', 'fk_p': 'Fattore sulla conducibilità termica delle polpette', 'fc_m': 'Fattore sulla capacità termica volumetrica della purea', 'fc_p': 'Fattore sulla capacità termica volumetrica delle polpette', 'fh': 'Fattore sul coefficiente di scambio superficiale'}
# grandezze: simbolo, descrizione, unita', soglia (%)
Q = {'T_P1': ('T_P1', 'Temperatura alla sonda P1', '°C', 2.5), 'T_P2': ('T_P2', 'Temperatura alla sonda P2', '°C', 2.5), 'T_P3': ('T_P3', 'Temperatura alla sonda P3', '°C', 2.5), 'T_max': ('T_max', 'Temperatura massima (nodo più caldo)', '°C', 5.0),
     'scarto_pol_pur': ('T̄_m − T̄_p', 'Differenza fra le temperature medie delle fasi', 'K', 2.5), 'T_media_polp': ('T̄_p', 'Temperatura media delle polpette', '°C', 2.5), 'T_media_pur': ('T̄_m', 'Temperatura media della purea', '°C', 2.5), 'E_in': ('E_in', 'Energia elettrica fornita', 'J', 2.5),
     'P': ('P', 'Potenza elettrica totale', 'W', 2.5), 'frac_P_p': ('Π_p', 'Frazione di potenza dissipata nelle polpette', '-', 2.5), 'q_media_polp': ('q̄_p', 'Sorgente media nelle polpette', 'W/m³', 2.5), 'q_media_pur': ('q̄_m', 'Sorgente media nella purea', 'W/m³', 2.5),
     'q_strati_z': ('q̄_z', 'Sorgente negli strati sopra e sotto le sfere', 'W/m³', 5.0), 'q_asse': ('q̄_ax', "Sorgente sull'asse della corrente", 'W/m³', 5.0), 'q_p99': ('q_99', '99° percentile della sorgente', 'W/m³', 5.0)}
QTXT = {k: v[0] for k, v in Q.items()}
STUDI = {'A': 'A · spaziale, elettrico', 'B': 'B · spaziale, accoppiato', 'C': 'C · temporale, accoppiato'}


def _leggi(nome, **kw): return pd.read_csv(os.path.join(DATA, nome), **kw)


@st.cache_data(show_spinner=False)
def convergenza():
    d = _leggi('convergenza_dettaglio.csv.gz'); d['sale'] = d.Ricetta.map(SALE); d['contrasto'] = d.Ricetta.map(CONTR); return d


@st.cache_data(show_spinner=False)
def livelli():
    d = _leggi('livelli_valori.csv.gz'); sp = d.id.str.split('__', expand=True); d['config'] = sp[1]; rv = sp[0].str.split('_', expand=True); d['ric'] = rv[0]; d['V'] = rv[1].str.replace('V', '').astype(int); return d


@st.cache_data(show_spinner=False)
def passo_temporale():
    d = _leggi('passo_temporale.csv.gz'); sp = d.id.str.split('__', expand=True); d['config'] = sp[1]; rv = sp[0].str.split('_', expand=True); d['ric'] = rv[0]; d['V'] = rv[1].str.replace('V', '').astype(int); return d


@st.cache_data(show_spinner=False)
def sens_locale():
    d = _leggi('sens_locale.csv.gz'); d['ric'] = d.id.str.split('_').str[0]; d['V'] = d.id.str.split('_').str[1].str.replace('V', '').astype(int); return d


@st.cache_data(show_spinner=False)
def _gl(nome):
    d = _leggi(nome); d['ric'] = d.caso.str.split('_').str[0]; d['V'] = d.caso.str.split('_').str[1].str.replace('V', '').astype(int); d['cfg'] = d.caso.str.split('__').str[1]; return d


def sobol(): return _gl('sobol.csv.gz')
def morris(): return _gl('morris.csv.gz')
def qualita(): return _gl('qualita.csv.gz')
def loc_glob(): return _gl('confronto_locale_globale.csv.gz')


@st.cache_data(show_spinner=False)
def riepilogo():
    d = _leggi('riepilogo_casi.csv.gz'); sp = d.id.str.split('__', expand=True); d['config'] = sp[1]; rv = sp[0].str.split('_', expand=True); d['ric'] = rv[0]; d['V'] = rv[1].str.replace('V', '').astype(int); return d


@st.cache_data(show_spinner=False)
def interfaccia():
    d = _leggi('confronto_interfaccia.csv.gz'); d['ric'] = d.caso.str.split('_').str[0]; return d


@st.cache_data(show_spinner=False)
def sens_confronto(): return _leggi('confronto_sensitivita.csv.gz')


@st.cache_data(show_spinner=False)
def intervalli(): return _leggi('fonti_intervalli_globali.csv')


@st.cache_data(show_spinner=False)
def verifica(nome): return _leggi(nome)


@st.cache_data(show_spinner=False)
def sperimentali():
    d = _leggi('dati_sperimentali.csv.gz'); d['run'] = ((d.t < d.t.shift()) | (d.cfg != d.cfg.shift()) | (d.V != d.V.shift()) | (d.NaCl != d.NaCl.shift())).cumsum()
    inv = {v: k for k, v in SALE.items()}; d['ric'] = d.NaCl.map(inv); return d


def sigma_T(ric, T): a, b = AB[ric]; return a * np.asarray(T, float) + b
def sigma_polp(T): return SIGMA_POLP[0] * np.asarray(T, float) + SIGMA_POLP[1]
