"""
Calibrazione e validazione del gemello digitale con simulazioni DIRETTE (nessun metamodello).

* Cella 99 x 54 x 66 mm, ricette A-E (sale 2.9, 5.7, 0, 0.725, 1.45%), conducibilita' dei registri del protocollo (dg.registro_ricette), mesh conforme.
* Condizioni iniziali dalle sonde: ogni polpetta parte dalla temperatura letta dalla propria sonda a t = 0; la purea dalla media delle sonde in purea
  (in Tris non ci sono sonde in purea: si assume la media delle tre polpette, IPOTESI).
* Parametri (moltiplicatori del valore nominale): k_sig (conducibilita' della purea), k_alfa (suo coefficiente di temperatura), f_c_m e f_c_p (capacita' termiche).
  sigma'(T) = k_sig * sigma20 * [1 + k_alfa * alfa * (T - 20)]
* Candidati annidati: M1 = (k_sig, f_c_m), M2 = + k_alfa, M3 = + f_c_p.
* Divisione dei dati come addestramento / test: validazione incrociata a K fold PER CONDIZIONE (configurazione, tensione, sale), stratificata per sale e tensione;
  si confrontano gli indici sul set di addestramento e su quello di test (RMSE, bias, MAE, R^2) e con il rumore minimo di misura.
Si confrontano gli INCREMENTI di temperatura T(t) - T(0) alle tre sonde.
"""
import os, json, time, pickle, hashlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares
import diagnostica_ohmico as dg
import protocollo_articolo as pa
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

SALE_RICETTA = {0.0: 'C', 0.725: 'D', 1.45: 'E', 2.9: 'A', 5.7: 'B'}
SFERE = {'Tris': [0, 1, 2], 'Symm': [0, 2], 'Asymm': [0, 1], 'Middle': [1], 'Side': [0]}     # posizioni occupate = sonde in polpetta
CANDIDATI = {'M1': ['k_sig', 'f_c_m'], 'M2': ['k_sig', 'k_alfa', 'f_c_m'], 'M3': ['k_sig', 'k_alfa', 'f_c_m', 'f_c_p']}
PRIOR = dict(k_sig=(1.0, 0.30), k_alfa=(1.0, 0.30), f_c_m=(1.0, 0.05), f_c_p=(1.0, 0.05))     # (media, scarto tipo): ASSUNTI
LIMITI = dict(k_sig=(0.2, 2.5), k_alfa=(0.2, 3.0), f_c_m=(0.7, 1.4), f_c_p=(0.7, 1.4))
TG = np.arange(0, 270.1, 5.0)


# ------------------------------------------------------------------------------------------------ dati
def carica_dati(path):
    d = pd.read_excel(path); d.columns = ['cfg', 'V', 'NaCl', 't', 'T1', 'T2', 'T3']
    d['run'] = ((d.t < d.t.shift()) | (d.cfg != d.cfg.shift()) | (d.V != d.V.shift()) | (d.NaCl != d.NaCl.shift())).cumsum()
    return d


def prove(d):
    """Una prova = una serie; condizioni iniziali dalle sonde a t = 0."""
    out = []
    for r, g in d.groupby('run'):
        cfg = g.cfg.iloc[0]; T = g[['T1', 'T2', 'T3']].values.astype(float); sf = SFERE[cfg]; pur = [i for i in range(3) if i not in sf]
        T0s = [float(T[0, i]) for i in sf]; T0m = float(np.mean([T[0, i] for i in pur])) if pur else float(np.mean(T0s))
        out.append(dict(run=int(r), cfg=cfg, V=float(g.V.iloc[0]), sale=float(g.NaCl.iloc[0]), t=g.t.values.astype(float), dT=T - T[0], T0m=T0m, T0s=T0s, ric=SALE_RICETTA[float(g.NaCl.iloc[0])],
                        pur_in_Tris=(not pur)))
    return out


def gruppi(runs, tol=1.5):
    """Simulazioni da eseguire: prove con stessa condizione e condizioni iniziali (purea e polpette) entro 'tol' K dal rappresentante del gruppo."""
    per = {}
    for r in runs: per.setdefault((r['cfg'], r['V'], r['sale']), []).append(r)
    out = []
    for k, rr in per.items():
        rr = sorted(rr, key=lambda x: x['T0m']); cl = []
        for x in rr:
            v = np.array([x['T0m']] + x['T0s'])
            for c in cl:
                if np.max(np.abs(v - c['v'])) <= tol: c['runs'].append(x); break
            else: cl.append(dict(v=v, runs=[x]))
        for i, c in enumerate(cl):
            m = np.mean([[x['T0m']] + x['T0s'] for x in c['runs']], axis=0)
            out.append(dict(id=f"{SALE_RICETTA[k[2]]}_{int(k[1])}V|{k[0]}|{i}", cond=f"{SALE_RICETTA[k[2]]}_{int(k[1])}V|{k[0]}", cfg=k[0], V=k[1], sale=k[2], ric=SALE_RICETTA[k[2]], T0m=float(m[0]), T0s=[float(x) for x in m[1:]], runs=c['runs']))
    return out


def stima_uD(gr):
    """u_D(DeltaT) = sqrt((a DeltaT)^2 + b^2) dalle ripetizioni dentro i gruppi (stesse condizioni iniziali)."""
    X, Y = [], []
    for g in gr:
        rr = g['runs']
        if len(rr) < 2: continue
        n = min(len(x['t']) for x in rr)
        for p in range(3):
            M = np.array([x['dT'][:n, p] for x in rr]); X += list(np.abs(M.mean(0))[12::6]); Y += list(M.std(0, ddof=1)[12::6])
    X, Y = np.array(X), np.array(Y); sol = np.linalg.lstsq(np.column_stack([X ** 2, np.ones_like(X)]), Y ** 2, rcond=None)[0]
    return dict(a=float(np.sqrt(max(sol[0], 1e-6))), b=float(max(np.sqrt(max(sol[1], 4e-4)), 0.05)), n=int(len(X)))


def uD(dT, ud): return np.sqrt((ud['a'] * dT) ** 2 + ud['b'] ** 2)


# ------------------------------------------------------------------------------------------------ modello diretto
_W = {}                                                    # cache per processo: contesti per configurazione e casi base


def _applica(p, th, ric_alfa):
    a0, b0 = p['matrice_a'], p['matrice_b']; s20 = 20 * a0 + b0; al = a0 / s20; ks, ka = th.get('k_sig', 1.0), th.get('k_alfa', 1.0)
    p['matrice_a'] = ks * s20 * ka * al; p['matrice_b'] = ks * s20 * (1 - 20 * ka * al)
    p['fc_matrice'] = th.get('f_c_m', 1.0); p['fc_polpetta'] = th.get('f_c_p', 1.0)
    return p


def _simula(task):
    cfg, ric, V, T0m, T0s, th, liv, dt, T_amb, cartella_mesh, fam = task
    k = ('ctx', cfg, liv, fam)
    if k not in _W:
        caso = dg.costruisci_caso(f'{ric}_{int(V)}V', cfg); o = pa.opzioni_art(dict(cartella_mesh=cartella_mesh)); _W[k] = pa.contesto(caso, fam, liv, o)
    ctx = _W[k]; caso = dg.costruisci_caso(f'{ric}_{int(V)}V', cfg); p = dict(caso['p'])
    _applica(p, th, None); p.update(T_iniziale=T0m, T_iniziale_polpetta=float(np.mean(T0s)), T_iniziale_sfere=list(T0s), T_amb=T_amb, dt=dt, durata=270.0, V_rms=float(V))
    serie, info = dg.simula_diag(ctx, p, 1e9, durata_max=270.0, t_min_run=270.0)
    col_t = 'tempo_s' if 'tempo_s' in serie.columns else serie.columns[0]; t = serie[col_t].values
    T = np.array([np.interp(TG, t, serie[f'T_P{i + 1}'].values) for i in range(3)]).T
    return T - np.array([T[0]])


class Simulatore:
    def __init__(self, liv=2, dt=10.0, T_amb=21.0, n_jobs=1, cartella_mesh='/tmp/mesh_cal', cache_file=None, famiglia='conforme'):
        self.liv, self.dt, self.T_amb, self.n_jobs, self.cm, self.cache_file, self.fam = liv, dt, T_amb, n_jobs, cartella_mesh, cache_file, famiglia
        self.cache = {}; self.n_sim = 0; self.t_sim = 0.0
        if cache_file and os.path.exists(cache_file):
            try: self.cache = pickle.load(open(cache_file, 'rb'))
            except Exception: self.cache = {}
        self.pool = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None

    def _prepara(self, cfg, ric, liv):
        """Genera in modo sequenziale (nel processo principale) la mesh conforme della configurazione: i processi in parallelo poi la leggono soltanto."""
        if not hasattr(self, '_pronte'): self._pronte = set()
        if self.fam == 'conforme' and (cfg, liv) not in self._pronte:
            import diagnostica_conforme as dgc
            o = pa.opzioni_art(dict(cartella_mesh=self.cm)); caso = dg.costruisci_caso(f'{ric}_40V', cfg); dgc.mesh_livello(caso, liv, pa.opz_conforme(o, caso['geo'])); self._pronte.add((cfg, liv))

    def _liv(self, g): return g.get('liv', self.liv)

    def _dt(self, g): return g.get('dt', self.dt)

    def _key(self, g, th): return (g['id'], round(g['T0m'], 2), tuple(round(x, 2) for x in g['T0s']), tuple(sorted((k, round(float(v), 5)) for k, v in th.items())), self._liv(g), self._dt(g), self.T_amb, self.fam)

    def calcola(self, grp, th, barra=None):
        """Incrementi simulati (31 tempi x 3 sonde) per ciascun gruppo, con i parametri th (dict)."""
        da = [g for g in grp if self._key(g, th) not in self.cache]
        for g in da: self._prepara(g['cfg'], g['ric'], self._liv(g))
        tasks = [(g['cfg'], g['ric'], g['V'], g['T0m'], g['T0s'], th, self._liv(g), self._dt(g), self.T_amb, self.cm, self.fam) for g in da]; t0 = time.time()
        if tasks:
            ris = list(self.pool.map(_simula, tasks)) if self.pool else [_simula(t) for t in tasks]
            for g, r in zip(da, ris): self.cache[self._key(g, th)] = r
            self.n_sim += len(tasks); self.t_sim += time.time() - t0
        return [self.cache[self._key(g, th)] for g in grp]

    def salva(self):
        if self.cache_file: pickle.dump(self.cache, open(self.cache_file, 'wb'))


# ------------------------------------------------------------------------------------------------ indici e calibrazione
def indici(S, D, w=None):
    S, D = np.asarray(S), np.asarray(D); e = S - D
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)), MAE=float(np.mean(np.abs(e))), R2=float(1 - np.sum(e ** 2) / np.sum((D - D.mean()) ** 2)), n=int(len(D)))


def coppie(grp, sims, ud, passo=2):
    """Vettori simulato, misurato, incertezza di misura e metadati, dai tempi 10 s in poi ogni 5*passo s."""
    S, D, U, M = [], [], [], []
    for g, sim in zip(grp, sims):
        for r in g['runs']:
            n = len(r['t']); idx = np.arange(2, n, passo)
            if r['t'][idx[-1]] != 5.0 * idx[-1]: idx = idx[5.0 * idx <= r['t'][-1]]
            for p in range(3):
                S += list(sim[idx, p]); d = r['dT'][idx, p]; D += list(d); U += list(uD(d, ud))
                M += [(g['cond'], g['V'], g['ric'], g['cfg'], 'polpetta' if p in SFERE[g['cfg']] else 'purea', f'P{p + 1}', float(tt)) for tt in r['t'][idx]]
    return np.array(S), np.array(D), np.array(U), pd.DataFrame(M, columns=['cond', 'V', 'ric', 'cfg', 'fase', 'sonda', 't'])


def calibra(grp, sim, ud, nomi, x0=None, max_nfev=8, verbose=0, callback=None):
    x0 = np.array([PRIOR[k][0] for k in nomi] if x0 is None else x0, float)
    def f(x):
        th = dict(zip(nomi, x)); S, D, U, _ = coppie(grp, sim.calcola(grp, th), ud); sim.salva()
        res = np.concatenate([(S - D) / U, [(x[i] - PRIOR[k][0]) / PRIOR[k][1] for i, k in enumerate(nomi)]])
        if callback is not None: callback(dict(zip(nomi, map(float, x))), float(np.sum(res ** 2)))
        return res
    lo, hi = [LIMITI[k][0] for k in nomi], [LIMITI[k][1] for k in nomi]
    r = least_squares(f, x0, bounds=(lo, hi), diff_step=0.03, max_nfev=max_nfev, x_scale=[PRIOR[k][1] for k in nomi], verbose=verbose)
    J = r.jac; cov = np.linalg.pinv(J.T @ J)
    return dict(theta=dict(zip(nomi, map(float, r.x))), sd=dict(zip(nomi, map(float, np.sqrt(np.diag(cov))))), nfev=int(r.nfev), costo=float(2 * r.cost), cov=[[float(v) for v in riga] for riga in cov], nomi=list(nomi))


def folds_per_condizione(grp, K, seed=20261104):
    """Assegna ogni CONDIZIONE (con tutte le sue prove) a un fold, stratificando per (sale, tensione)."""
    rng = np.random.default_rng(seed); cond = sorted({g['cond'] for g in grp}); info = {c: next(g for g in grp if g['cond'] == c) for c in cond}; strati = {}
    for c in cond: strati.setdefault((info[c]['sale'], info[c]['V']), []).append(c)
    f = {}; k = 0
    for s, cs in sorted(strati.items()):
        rng.shuffle(cs)
        for c in cs: f[c] = k % K; k += 1
    return f


# ------------------------------------------------------------------------------------------------ mesh e passo suggeriti dal protocollo
def livelli_h(nodi_grossa=800, fattore=2 ** 1.5, n=5, geom=(0.099, 0.054, 0.066)):
    """Dimensione h = (V/N)^(1/3) [mm] dei livelli di mesh del protocollo (stessa definizione per le due famiglie)."""
    V = geom[0] * geom[1] * geom[2]
    return [((V / (nodi_grossa * fattore ** k)) * 1e9) ** (1 / 3) for k in range(n)]


def suggerimenti_protocollo(path, colonna_h='h_temperature_mm', colonna_dt='C_dt_min', famiglia='conforme'):
    """Legge il foglio 'riepilogo_casi' dei risultati del protocollo (famiglia conforme) e restituisce {(ricetta, tensione, configurazione): (livello, dt)}.
    Livello: il piu' grossolano fra i cinque con h <= 1.1 * h suggerito (quello suggerito per le sole temperature, 'h_temperature_mm'); passo: 'C_dt_min'."""
    if os.path.isdir(path): path = os.path.join(path, famiglia, 'risultati_completi.xlsx')
    R = pd.read_excel(path, sheet_name='riepilogo_casi'); hk = livelli_h(); out = {}
    for _, r in R.iterrows():
        ric_v, cfg = str(r['id']).split('__')[:2]; ric, v = ric_v.split('_'); h_s = float(r[colonna_h]); dt = float(r[colonna_dt])
        ok = [k for k in range(len(hk)) if hk[k] <= 1.1 * h_s]; liv = min(ok) if ok else len(hk) - 1
        out[(ric, float(v.replace('V', '')), cfg)] = (int(liv), dt)
    return out


def assegna_suggeriti(grp, sug, liv_default, dt_default):
    """Assegna a ogni gruppo il livello e il passo suggeriti per la sua (ricetta, tensione, configurazione); se mancano, i valori di default."""
    n_ok = 0
    for g in grp:
        k = (g['ric'], g['V'], g['cfg'])
        if k in sug: g['liv'], g['dt'] = sug[k]; n_ok += 1
        else: g['liv'], g['dt'] = liv_default, dt_default
    return n_ok
