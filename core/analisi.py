"""Analisi numeriche sopra i risultati del protocollo e analisi dirette (verifica rapida, sensitivita' locale in diretta)."""
import numpy as np
import pandas as pd
import streamlit as st
from . import dati
from . import motore as mo
dg = mo.dg


def stimatore(h, f, metodo='gci3'):
    """Limite, ordine, regime e incertezza di discretizzazione (funzione del protocollo)."""
    return dg.incertezza_discretizzazione(np.asarray(h, float), np.asarray(f, float), metodo)


@st.cache_data(show_spinner=False)
def dt_tabella(soglia=2.5):
    """Errore dovuto al passo temporale rispetto al limite per dt -> 0 (minimi quadrati), ordine osservato e passo richiesto, per T_P1 ed E_in."""
    P = dati.passo_temporale(); out = []
    for (fam, cid, q), g in P[P.grandezza.isin(['T_P1', 'E_in'])].groupby(['famiglia', 'id', 'grandezza']):
        g = g.sort_values('dt'); x = g.dt.values.astype(float); y = g.valore.values.astype(float); off = 20.0 if q == 'T_P1' else 0.0
        est = stimatore(x, y, 'ls'); f0 = est['f_best'] - off; yy = y - off; d = dict(zip(x, yy)); sc = abs(f0)
        if not all(k in d for k in (5.0, 10.0, 20.0, 40.0)): continue
        e = {t: 100 * abs(d[t] - f0) / sc for t in (5.0, 10.0, 20.0, 40.0)}; den = abs(d[10.0] - d[5.0]); p = float(np.log2(abs(d[20.0] - d[10.0]) / den)) if den > 0 else np.nan
        pu = float(np.clip(p, 0.8, 2.0)) if np.isfinite(p) else 1.0; a = abs(d[5.0] - f0) / 5.0 ** pu; dtreq = (soglia / 100 * sc / a) ** (1 / pu) if a > 0 else np.nan
        ric, cfg = cid.split('__'); r0, V = ric.split('_'); out.append(dict(famiglia=fam, ric=r0, V=int(V.replace('V', '')), config=cfg, grandezza=q, limite=f0, ordine=p, dt_richiesto=dtreq, **{f'e{int(k)}': v for k, v in e.items()}))
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def errore_livelli():
    """Errore ai livelli di mesh rispetto al limite comune (minimi quadrati della mesh conforme dello stesso caso), in % della scala."""
    D = dati.convergenza(); L = dati.livelli(); ref = D[D.Famiglia == 'conforme'].set_index(['Ricetta', 'Tensione_V', 'Configurazione', 'Grandezza', 'Studio'])[['Limite_LS', 'Scala', 'Soglia_pc']]
    L = L.merge(ref.reset_index(), left_on=['ric', 'V', 'config', 'grandezza', 'studio'], right_on=['Ricetta', 'Tensione_V', 'Configurazione', 'Grandezza', 'Studio'], how='inner')
    L['err'] = 100 * (L.valore - L.Limite_LS).abs() / L.Scala; return L[['famiglia', 'studio', 'grandezza', 'ric', 'V', 'config', 'livello', 'n_nodi', 'h_rep_mm', 'err', 'Soglia_pc']]


def verifica_rapida(ric, V, conf, famiglie, livelli, dt, durata, T0=20.0, T_amb=21.0, avanz=None):
    """Raffinamento della mesh in diretta: valori finali di alcune grandezze ai livelli indicati, per le famiglie scelte."""
    righe = []; n = len(famiglie) * len(livelli); k = 0
    for fam in famiglie:
        for lv in livelli:
            ris = mo.simula(ric, V, conf, fam, lv, T0=T0, T_amb=T_amb, dt=dt, durata=durata); se = ris['serie']; r0 = se.iloc[0]; rf = se.iloc[-1]
            righe.append(dict(famiglia=fam, livello=lv, n_nodi=ris['info']['n_nodi'], h_mm=ris['info']['h_rep'] * 1000, tempo_s=ris['tempo'], T_P1=rf.T_P1 - r0.T_P1, T_P2=rf.T_P2 - r0.T_P2, T_P3=rf.T_P3 - r0.T_P3, T_max=rf.T_max - r0.T_max, E_in=rf.E_in, frac_P_p=r0.frac_P_p, q_media_polp=r0.q_media_polp, P=r0.P))
            k += 1
            if avanz: avanz(k / n)
    return pd.DataFrame(righe)


def sensibilita_diretta(ric, V, conf, famiglia, livello, dt, durata, delta=0.10, T0=20.0, T_amb=21.0, avanz=None):
    """Sensitivita' locale in diretta: differenze in avanti (+delta) di ciascun parametro sulle temperature alle sonde (tempi ogni 10 s). Restituisce (tabella, matrice X scalata, tempi)."""
    tempi = np.arange(10.0, durata + 1e-9, 10.0); base = mo.simula(ric, V, conf, famiglia, livello, T0=T0, T_amb=T_amb, dt=dt, durata=durata)
    def sonde(r): se = r['serie']; return np.vstack([np.interp(tempi, se.t.values, se[f'T_P{i + 1}'].values - se[f'T_P{i + 1}'].values[0]) for i in range(3)])
    S0 = sonde(base); cols = {}; n = len(dati.PAR)
    for k, par in enumerate(dati.PAR):
        r = mo.simula(ric, V, conf, famiglia, livello, T0=T0, T_amb=T_amb, dt=dt, durata=durata, mult={par: 1 + delta}); cols[par] = ((sonde(r) - S0) / delta).ravel()
        if avanz: avanz((k + 1) / n)
    X = np.column_stack([cols[p] for p in dati.PAR]); return X, tempi, S0
