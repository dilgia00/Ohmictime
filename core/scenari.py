"""Scenari di processo, ottimizzazione della tensione e propagazione dell'incertezza dei parametri (simulazioni dirette del motore FEM)."""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import qmc
from . import dati
from . import motore as mo

LIM_T = 90.0                                                                  # il modello non include l'ebollizione
PARAM_UNC = ['sig_m20', 'alfa_m', 'sig_p20', 'alfa_p', 'fc_m', 'fc_p']
SEMI = {'sig_m20': 0.10, 'sig_p20': 0.10, 'alfa_m': 0.41, 'alfa_p': 0.41, 'fk_m': 0.12, 'fk_p': 0.12, 'fc_m': 0.08, 'fc_p': 0.08, 'fh': 0.35}   # semiampiezze relative (intervalli dell'analisi globale)


def tempo_a_target(t, T, target):
    """Primo istante in cui T raggiunge il target (interpolazione lineare); NaN se non lo raggiunge."""
    t, T = np.asarray(t, float), np.asarray(T, float); i = np.where(T >= target)[0]
    if len(i) == 0: return np.nan
    k = i[0]
    return float(t[0]) if k == 0 else float(t[k - 1] + (target - T[k - 1]) * (t[k] - t[k - 1]) / max(T[k] - T[k - 1], 1e-12))


def sintesi_serie(se, target=None, sonda='fredda'):
    """Indicatori di un caso simulato: incrementi finali, T_max, energia, potenza media e tempo per raggiungere un target alla sonda piu' fredda."""
    t = se.t.values; Tp = np.vstack([se[f'T_P{i}'].values for i in (1, 2, 3)]); fredda = int(np.argmin(Tp[:, -1])); d = dict(T_P1=Tp[0, -1], T_P2=Tp[1, -1], T_P3=Tp[2, -1], T_max=float(se.T_max.iloc[-1]), E_kJ=float(se.E_in.iloc[-1]) / 1000.0, P_media_W=float(se.P.mean()), sonda_fredda=f'P{fredda + 1}', T_fredda=float(Tp[fredda, -1]), margine_limite=LIM_T - float(se.T_max.iloc[-1]))
    if target is not None: d['t_target_s'] = tempo_a_target(t, Tp[fredda], target); d['t_limite_s'] = tempo_a_target(t, se.T_max.values, LIM_T)
    return d


def simula_scenario(sc, famiglia, livello, dt=10.0, durata=270.0, T_amb=21.0, mult=None):
    """Una simulazione di uno scenario: sc = dict(ric, V, config, T0)."""
    return mo.simula(sc['ric'], sc['V'], sc['config'], famiglia, livello, T0=sc.get('T0', 20.0), T_amb=T_amb, dt=dt, durata=durata, mult=mult)


def lhs_moltiplicatori(n, params, seed=1234, semi=None):
    """Campione a ipercubo latino dei moltiplicatori dei parametri, uniformi in [1 - d, 1 + d]."""
    semi = semi or SEMI; X = qmc.LatinHypercube(d=len(params), seed=seed).random(n)
    return pd.DataFrame({p: 1 + semi[p] * (2 * X[:, j] - 1) for j, p in enumerate(params)})


def ensemble(sc, famiglia, livello, M, dt=20.0, durata=270.0, T_amb=21.0, avanz=None):
    """Una simulazione per ciascuna riga di M (moltiplicatori). Restituisce (tempi, T[n, 3, nt], Tmax[n, nt], E[n])."""
    out = []; tempi = None
    for k, (_, r) in enumerate(M.iterrows()):
        ris = simula_scenario(sc, famiglia, livello, dt, durata, T_amb, mult={p: float(r[p]) for p in M.columns}); se = ris['serie']
        if tempi is None: tempi = se.t.values
        out.append((np.vstack([se[f'T_P{i}'].values for i in (1, 2, 3)]), se.T_max.values, float(se.E_in.iloc[-1])))
        if avanz: avanz((k + 1) / len(M))
    return tempi, np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out])


def coefficienti_standardizzati(M, y):
    """Coefficienti di regressione standardizzati (SRC) e R^2 della regressione lineare di y sui moltiplicatori."""
    X = (M.values - M.values.mean(0)) / np.where(M.values.std(0) > 0, M.values.std(0), 1.0); yy = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0); b, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), X], yy, rcond=None); res = yy - np.c_[np.ones(len(X)), X] @ b
    return dict(zip(M.columns, b[1:])), float(1 - np.sum(res ** 2) / max(np.sum(yy ** 2), 1e-12))


def prove_corrispondenti(ric, V, cfg):
    """Prove sperimentali con la stessa ricetta, tensione e configurazione: lista di (t, T1, T2, T3)."""
    d = dati.sperimentali(); x = d[(d.ric == ric) & (d.V == V) & (d.cfg == cfg)]
    return [(g.t.values, g[['T1', 'T2', 'T3']].values) for _, g in x.groupby('run')]
