"""Dati sperimentali, raggruppamento per condizioni iniziali, modello del rumore e utilita' di calibrazione (sopra engine/calibrazione_protocollo.py)."""
import numpy as np
import pandas as pd
import streamlit as st
from . import dati
from . import motore as mo          # imposta il percorso del motore
import calibrazione_protocollo as cp


@st.cache_data(show_spinner=False)
def prove_lista():
    d = dati.sperimentali()[['cfg', 'V', 'NaCl', 't', 'T1', 'T2', 'T3', 'run']]
    return cp.prove(d)


@st.cache_data(show_spinner=False)
def gruppi_lista(tol=2.0): return cp.gruppi(prove_lista(), tol)


@st.cache_data(show_spinner=False)
def uD_stima(tol=1.5): return cp.stima_uD(cp.gruppi(prove_lista(), tol))


def uD(dT, ud): return cp.uD(dT, ud)


@st.cache_data(show_spinner=False)
def ripetizioni(tol=1.5):
    """Deviazione standard fra ripetizioni in funzione dell'incremento medio (alle sonde, ogni 30 s)."""
    X, Y, M = [], [], []
    for g in cp.gruppi(prove_lista(), tol):
        rr = g['runs']
        if len(rr) < 2: continue
        n = min(len(x['t']) for x in rr)
        for p in range(3):
            A = np.array([x['dT'][:n, p] for x in rr]); m = np.abs(A.mean(0))[12::6]; s = A.std(0, ddof=1)[12::6]; X += list(m); Y += list(s); M += [(g['ric'], g['V'], g['cfg'])] * len(m)
    return pd.DataFrame(dict(dT=X, sd=Y, ric=[m[0] for m in M], V=[m[1] for m in M], cfg=[m[2] for m in M]))


def catalogo_prove():
    rows = []
    for r in prove_lista():
        rows.append(dict(run=r['run'], ric=r['ric'], V=int(r['V']), cfg=r['cfg'], sale=r['sale'], T0_purea=round(r['T0m'], 1), T0_sfere=', '.join(f'{x:.1f}' for x in r['T0s']), n=len(r['t']), t_max=float(r['t'][-1]),
                         dT_P1=float(r['dT'][-1, 0]), dT_P2=float(r['dT'][-1, 1]), dT_P3=float(r['dT'][-1, 2])))
    return pd.DataFrame(rows)


def prova(run):
    for r in prove_lista():
        if r['run'] == run: return r


# ------------------------------------------------------------------------------------------------ calibrazione e divisione dei dati
MODELLI = {'M1': ('Base: σ della purea e capacità termica della purea', cp.CANDIDATI['M1']), 'M2': ('M1 più il coefficiente di temperatura della purea', cp.CANDIDATI['M2']), 'M3': ('M2 più la capacità termica delle polpette', cp.CANDIDATI['M3'])}
PARAM_TXT = {'k_sig': 'k_σ · moltiplicatore di σ₂₀ della purea', 'k_alfa': 'k_α · moltiplicatore del coefficiente di temperatura', 'f_c_m': 'f_c,m · fattore sulla capacità termica della purea', 'f_c_p': 'f_c,p · fattore sulla capacità termica delle polpette'}


def nuovo_simulatore(famiglia, livello, dt, T_amb):
    return cp.Simulatore(liv=int(livello), dt=float(dt), T_amb=float(T_amb), n_jobs=1, cartella_mesh=mo.MESHDIR, famiglia=famiglia)


def esegui_divisione(train, test, nomi, sim, ud, max_nfev=5, callback=None, etichetta='M1'):
    """Calibra sul set di addestramento e valuta su addestramento e test (con i parametri nominali come riferimento). Stesso formato dei risultati del notebook."""
    import time
    t0 = time.time(); cal = cp.calibra(train, sim, ud, nomi, max_nfev=max_nfev, callback=callback); th = cal['theta']
    Str, Dtr, Utr, Mtr = cp.coppie(train, sim.calcola(train, th), ud); pred = [Mtr.assign(S=Str, D=Dtr, U=Utr, set='train')]; r = dict(cand=etichetta, theta=th, sd=cal['sd'], cov=cal.get('cov'), nomi=cal.get('nomi'), nfev=cal['nfev'], gruppi_train=len(train), gruppi_test=len(test), train=cp.indici(Str, Dtr), rumore_min_train=float(np.sqrt(np.mean(Utr ** 2))))
    S0t, D0t, _, _ = cp.coppie(train, sim.calcola(train, {}), ud); r['train_nominale'] = cp.indici(S0t, D0t)
    if test:
        Ste, Dte, Ute, Mte = cp.coppie(test, sim.calcola(test, th), ud); S0, D0, _, M0 = cp.coppie(test, sim.calcola(test, {}), ud); pred.append(Mte.assign(S=Ste, D=Dte, U=Ute, set='test')); pred.append(M0.assign(S=S0, D=D0, U=Ute, set='test_nominale'))
        r.update(test=cp.indici(Ste, Dte), test_nominale=cp.indici(S0, D0), rumore_min_test=float(np.sqrt(np.mean(Ute ** 2))), condizioni_test=sorted({g['cond'] for g in test}))
    r['minuti'] = (time.time() - t0) / 60; r['n_sim'] = sim.n_sim; return r, pd.concat(pred, ignore_index=True)


def valuta_parametri(gruppi, th, sim, ud):
    """Valutazione di un insieme di parametri su gruppi di condizioni, con il riferimento nominale."""
    S, D, U, M = cp.coppie(gruppi, sim.calcola(gruppi, th), ud); S0, D0, _, M0 = cp.coppie(gruppi, sim.calcola(gruppi, {}), ud)
    return pd.concat([M.assign(S=S, D=D, U=U, set='test'), M0.assign(S=S0, D=D0, U=U, set='test_nominale')], ignore_index=True)


def importa(file_json, file_csv=None):
    """Legge i file prodotti dal notebook di calibrazione (cv_<modello>_fold<k>.json e previsioni_<modello>_fold<k>.csv)."""
    import json
    r = json.load(file_json) if hasattr(file_json, 'read') else json.loads(file_json); pred = None
    if file_csv is not None:
        p = pd.read_csv(file_csv); p['set'] = p['set'].replace({'test': 'test'}); pred = p
    return r, pred
