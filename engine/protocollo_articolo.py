"""
Protocollo per l'articolo scientifico: stesso studio su ENTRAMBE le discretizzazioni (griglia strutturata con interfaccia diffusa e
mesh conforme con interfaccia netta), con famiglie ARMONIZZATE e controlli pensati per resistere alle critiche dei revisori.

Armonizzazione
  * stesso NUMERO DI NODI a ogni livello per i due tipi di mesh (5 livelli, nodi x 2^1.5 a ogni livello, cioe' h ridotta di sqrt(2)); h = (V/N_nodi)^(1/3) con la stessa definizione per entrambe;
  * STESSO tempo di valutazione per le due famiglie (pilota condiviso: il minore fra i due);
  * stesso passo temporale, stesse mesh per lo studio del passo, stessa mesh e durata per la sensitivita'.

Controlli (vedi tabella_critiche()):
  verifica del codice con soluzioni analitiche; errore iterativo; validazione degli stimatori di incertezza con una mesh piu' fine
  (leave-finest-out); indipendenza della sensitivita' da delta, rumore e mesh; confronto diretto fra le due famiglie;
  riproducibilita' (ambiente, versioni, impronte dei file).
Richiede diagnostica_ohmico.py, diagnostica_conforme.py, multifase_engine.py, gmsh e meshio.
"""
import datetime
import hashlib
import json
import logging
import os
import platform
import sys
import time
import zlib

import numpy as np
import pandas as pd

logging.getLogger('skfem').setLevel(logging.ERROR)
import diagnostica_conforme as dgc          # noqa: E402
import diagnostica_ohmico as dg             # noqa: E402
import multifase_engine as me               # noqa: E402

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:                                    # senza tqdm si lavora senza barra
    _tqdm = None
_BARRA = dict(bar=None, sims=0, tot_sims=0)


def _fase(txt):
    """Aggiorna la fase mostrata accanto alla barra di avanzamento (se presente)."""
    b = _BARRA['bar']
    if b is not None:
        try:
            b.set_postfix_str(txt, refresh=True)
        except Exception:
            pass


FAMIGLIE = ('strutturata', 'conforme')

OPZ_ART = dict(
    livelli=5, nodi_grossa=800, fattore_nodi=2 ** 1.5,    # nodi-obiettivo per livello: 800, 2263, 6400, 18102, 51200 (uguali per le due famiglie; h ridotta di sqrt(2) a ogni livello)
    kappa_lc=1.9, tolleranza_nodi=0.06, rapporto_lci=0.5,   # conforme: dimensione globale iniziale = kappa (V/N)^(1/3), corretta fino a ottenere i nodi-obiettivo (+/-6%); all'interfaccia meta'
    max_nodi_strutturata=60000, max_nodi_conforme=60000, min_livelli=5,        # tetti uguali per le due famiglie: i 5 livelli sono sempre eseguiti
    livello_dt=(1, 3),                                      # studio del passo su due mesh (indipendenza dal passo-mesh)
    livello_it=2,                                           # mesh per il test dell'errore iterativo
    livelli_S=(1,), livelli_S_approfonditi=(1, 2), delta_check=0.05, rumori=(0.1, 0.25, 0.5, 1.0),
    rtol_test=(1e-4, 1e-6, 1e-8, 1e-12),
    configurazioni_approfondite=('Tris',),                  # casi con verifiche estese (sensitivita' su due mesh)
    esegui_iterativo=True, esegui_S=True,
    # sensitivita' globale (Morris + indici di Sobol da metamodello polinomiale): intervalli PROVVISORI, da giustificare con fonti prima dell'uso nel manoscritto
    esegui_globale=True, globale_famiglie=('conforme',), globale_configurazioni=('Tris', 'Symm', 'Asymm', 'Middle', 'Side'), livello_globale=1,
    morris_r=15, morris_p=4, lhs_n=200, globale_seed=20261031, globale_scale=(1.0,), globale_scale_config={'Tris': (1.0, 2.0)}, sobol_boot=200,
    # semiampiezze relative (distribuzione uniforme) = sqrt(3) x scarto tipo relativo di letteratura, cosi' da conservarne la deviazione standard (vedi tabella_fonti_intervalli())
    globale_ampiezza=dict(sig_m20=0.10, alfa_m=0.41, sig_p20=0.10, alfa_p=0.41, fk_m=0.12, fk_p=0.12, fc_m=0.08, fc_p=0.08, fh=0.35),
    cartella_mesh='meshes_conformi', tentativi_mesh=3,
)


def opzioni_art(opz=None):
    o = dict(dg.OPZ_DEFAULT); o.update(dgc.OPZ_CONF); o.update(OPZ_ART); o.update(opz or {})
    o['riferimento_netto'] = 'no'
    return o


# ---------------------------------------------------------------------------------------------------------
# Famiglie di mesh armonizzate
# ---------------------------------------------------------------------------------------------------------
def nodi_obiettivo(k, o):
    return o['nodi_grossa'] * o['fattore_nodi'] ** k


def n_strutturata(geo, k, o):
    """Intervalli (nx, ny, nz) della griglia strutturata con passo uniforme e numero di nodi piu' vicino al nodi-obiettivo del livello k."""
    N = nodi_obiettivo(k, o); dims = (geo['Lx'], geo['Ly'], geo['Lz']); h0 = (dims[0] * dims[1] * dims[2] / N) ** (1.0 / 3.0); best = None
    for s in np.linspace(0.7, 1.4, 561):
        n = tuple(max(2, int(round(L / (h0 * s)))) for L in dims); err = abs((n[0] + 1) * (n[1] + 1) * (n[2] + 1) / N - 1.0)
        if best is None or err < best[0] - 1e-12: best = (err, n)
    return best[1]


def opz_conforme(o, geo):
    oc = dict(o)
    oc.update(nodi_obiettivo=[nodi_obiettivo(k, o) for k in range(o['livelli'])], livelli_conf=o['livelli'])
    return oc


def contesto(caso, fam, k, o):
    """Contesto di calcolo del livello k per la famiglia indicata."""
    if fam == 'strutturata':
        ctx = dg.Contesto(caso['geo'], n_strutturata(caso['geo'], k, o), caso['p'], costruisci_filtro=False)
        ctx.n_tet = int(ctx.mesh.t.shape[1]); ctx.livello = k
        return ctx
    return dgc.contesto_conforme(caso, k, opz_conforme(o, caso['geo']))


def livelli_ammessi(caso, fam, o):
    """Livelli eseguibili: tetto ai nodi, ma almeno 'min_livelli'."""
    if fam == 'strutturata':
        ok = []
        for k in range(o['livelli']):
            n = n_strutturata(caso['geo'], k, o)
            if (n[0] + 1) * (n[1] + 1) * (n[2] + 1) <= o['max_nodi_strutturata'] or len(ok) < o['min_livelli']:
                ok.append(k)
        return ok
    return list(range(o['livelli']))                       # per la conforme il tetto e' applicato dopo la generazione


# ---------------------------------------------------------------------------------------------------------
# Tempo di valutazione condiviso
# ---------------------------------------------------------------------------------------------------------
def tempo_valutazione_condiviso(caso, o, base):
    """Pilota su entrambe le famiglie (livello 1, passo 5 s): t_val = minimo dei due tempi in cui T_max raggiunge il limite (arrotondato a 5 s)."""
    os.makedirs(os.path.join(base, 'pilota'), exist_ok=True)
    f = os.path.join(base, 'pilota', caso['id'] + '.json')
    if os.path.exists(f):
        return json.load(open(f))['t_val']
    t_fix = o['t_fisso']; cand = {}
    for fam in FAMIGLIE:
        ctx = contesto(caso, fam, min(1, o['livelli'] - 1), o)
        pp = dict(caso['p']); pp['dt'] = 5.0
        _, inf = dg.simula_diag(ctx, pp, 1e9, T_crit=float(o['T_limite_valutazione']), durata_max=t_fix + 10, t_min_run=0.0)
        cand[fam] = None if inf['t_boil'] is None or inf['t_boil'] >= t_fix else float(max(30.0, np.floor(inf['t_boil'] / 5.0) * 5.0))
        del ctx
    vals = [v for v in cand.values() if v is not None]
    t_val = min(vals) if vals else float(t_fix)
    json.dump(dict(t_val=t_val, candidati=cand), open(f, 'w'))
    return t_val


# ---------------------------------------------------------------------------------------------------------
# Sensitivita': sorgente delle temperature alle sonde su un contesto qualsiasi, selezione con rumore variabile
# ---------------------------------------------------------------------------------------------------------
def seleziona_da_X(X, sigma, se_max):
    norme = np.sqrt((X ** 2).mean(axis=0)); sel = []
    for j in np.argsort(-norme):
        if norme[j] <= 0 or not np.isfinite(norme[j]):
            continue
        prova = sel + [j]; Xt = X[:, prova]
        try:
            se_t = sigma * np.sqrt(np.abs(np.diag(np.linalg.inv(Xt.T @ Xt))))
        except np.linalg.LinAlgError:
            continue
        if np.all(se_t <= se_max):
            sel = prova
    return [dg.PARAMETRI_CAL[j] for j in sel]


def sensibilita_ctx(caso, ctx, o, dur_S, delta):
    geo, p = caso['geo'], caso['p']
    nomi = sorted(ctx.nomi_sonde); fase_p = dict(zip(ctx.nomi_sonde, ctx.Pr @ ctx.phi_n))

    def sonde_fn(theta, pb, tempi):
        pp = dg.applica_param(pb, theta); pp['dt'] = float(o['dt_S'])
        serie, _ = dg.simula_diag(ctx, pp, 1e9, durata_max=dur_S + 2 * o['dt_S'], t_min_run=dur_S)
        return np.vstack([np.interp(tempi, serie['t'].values, serie['T_' + n].values) for n in nomi]), {n: float(fase_p[n]) for n in nomi}
    return dg.sensibilita_locale(geo, p, None, o['dt_S'], dur_S, delta, o['passo_misura'], o['sigma_misura'], o['se_max'], sonde_fn=sonde_fn)


# ---------------------------------------------------------------------------------------------------------
# Analisi di un caso su una famiglia
# ---------------------------------------------------------------------------------------------------------
def analizza_famiglia(caso, fam, cartella, opz, base_pilota):
    o = opzioni_art(opz)
    cid, geo, p = caso['id'], caso['geo'], caso['p']
    dirc = os.path.join(cartella, cid); os.makedirs(dirc, exist_ok=True)
    t0 = time.time(); dt = o['dt']; T0c = p['T_iniziale']
    riep = dict(id=cid, ricetta=caso['ricetta'], config=caso['config'], cella=caso['cella'], sigma_m20=p['_sig_m20'], alfa_m=p['_alfa_m'], mesh_tipo=fam)
    t_eval = tempo_valutazione_condiviso(caso, o, base_pilota)
    riep.update(t_valutazione=t_eval, t_limitato_ebollizione=bool(t_eval < o['t_fisso']), t_fisso=o['t_fisso'])
    # ---- A e B ----
    rA, rB = [], []
    liv_tot = len(livelli_ammessi(caso, fam, o))
    for k in livelli_ammessi(caso, fam, o):
        _fase(f'mesh {k + 1}/{liv_tot}: costruzione e calcolo')
        ctx = contesto(caso, fam, k, o); N = int(ctx.basis.N)
        if fam == 'conforme' and N > o['max_nodi_conforme'] and len(rA) >= o['min_livelli']:
            del ctx; break
        e = dg.solo_elettrico(ctx, p); e.update(h_rep_mm=ctx.h_rep * 1000, n_nodi=N, n_tet=ctx.n_tet, livello=k, nodi_obiettivo=nodi_obiettivo(k, o)); rA.append(e)
        serie, info = dg.simula_diag(ctx, p, 1e9, durata_max=t_eval + 2 * dt, t_min_run=t_eval)
        tf = dg.valori_a_tempo(serie, t_eval); tf.update(h_rep_mm=ctx.h_rep * 1000, n_nodi=N, n_tet=ctx.n_tet, livello=k); rB.append(tf)
        riep['chiusura_max'] = max(riep.get('chiusura_max', 0.0), float(serie.chiusura_energia.abs().max()))
        del ctx
    dfA, dfB = pd.DataFrame(rA), pd.DataFrame(rB)
    dfA.to_csv(os.path.join(dirc, 'A.csv'), index=False); dfB.to_csv(os.path.join(dirc, 'B_tempo_fisso.csv'), index=False)
    riep.update(A_ident_max=float((dfA.P_meno_VI.abs() / dfA.P).max()), A_sbil_max=float(dfA.sbilancio_I.max()), A_griglie=len(dfA), B_griglie=len(dfB), A_nodi_max=int(dfA.n_nodi.max()))
    tol = {g: o['tol_abs_C'] for g in dg._QOI_T}
    # ---- C: passo temporale su DUE mesh ----
    for ic, kC in enumerate(o['livello_dt']):
        _fase(f'passo temporale (mesh {ic + 1}/{len(o["livello_dt"])})')
        kC = min(kC, len(dfA) - 1)
        ctxC = contesto(caso, fam, kC, o); dts = [dt * 4, dt * 2, dt, dt / 2]; righeC = []

        def _un_dt(d_):
            pp = dict(p); pp['dt'] = float(d_)
            serie, info = dg.simula_diag(ctxC, pp, 1e9, durata_max=t_eval + 2 * float(d_), t_min_run=t_eval)
            tf = dg.valori_a_tempo(serie, t_eval); tf.update(dt=d_, stato=info['stato'], livello=kC, n_nodi=int(ctxC.basis.N)); righeC.append(tf)
        for d_ in dts:
            _un_dt(d_)
        extra = 0
        while True:
            dfC = pd.DataFrame(righeC)
            gC = [g for g in ['T_media_polp', 'T_media_pur', 'scarto_pol_pur', 'T_P1', 'T_P2', 'T_P3', 'E_in'] if g in dfC.columns and np.all(np.isfinite(dfC[g].values))]
            riseC = float(dfC['T_max'].values[np.argmin(dfC['dt'].values)] - T0c)
            vC = dg.valuta_convergenza(dfC, dfC['dt'].values / 1000.0, gC, ['T_max'], o, {g: riseC for g in dg._QOI_T}, tol)
            if vC['h_ess'] is not None or extra >= o.get('dt_livelli_extra', 3):
                break
            _un_dt(float(dfC['dt'].min()) / 2.0); extra += 1
        dfC.to_csv(os.path.join(dirc, 'C.csv' if ic == 0 else f'C_mesh{ic + 1}.csv'), index=False)
        if ic == 0:
            riep.update(C_dt_livelli_extra=extra, C_nodi=int(ctxC.basis.N))
        del ctxC
    # ---- errore iterativo ----
    if o['esegui_iterativo']:
        _fase('errore iterativo')
        kI = min(o['livello_it'], len(dfA) - 1); ctxI = contesto(caso, fam, kI, o); righeI = []
        r0e, r0t = me.RTOL_ELETTRICO, me.RTOL_TERMICO
        try:
            for r in o['rtol_test']:
                me.RTOL_ELETTRICO, me.RTOL_TERMICO = r, r
                serie, _ = dg.simula_diag(ctxI, p, 1e9, durata_max=t_eval + 2 * dt, t_min_run=t_eval)
                tf = dg.valori_a_tempo(serie, t_eval); tf.update(rtol=r, n_nodi=int(ctxI.basis.N), solutore_el='AMG-CG' if (me.HA_PYAMG and ctxI.basis.N > 3000) else 'diretto'); righeI.append(tf)
        finally:
            me.RTOL_ELETTRICO, me.RTOL_TERMICO = r0e, r0t
        pd.DataFrame(righeI).to_csv(os.path.join(dirc, 'iter.csv'), index=False); del ctxI
    # ---- sensitivita': stessa mesh (livello), stessa durata, due delta, rumore variabile, eventualmente due mesh ----
    if o['esegui_S']:
        appro = caso['config'] in o['configurazioni_approfondite']
        livS = list(o['livelli_S_approfonditi'] if appro else o['livelli_S'])
        dur_S = float(min(o['durata_misure'], np.floor(t_eval / o['passo_misura']) * o['passo_misura']))
        for i_l, kS in enumerate(livS):
            _fase(f'sensitivita locale (mesh {i_l + 1}/{len(livS)})')
            kS = min(kS, len(dfA) - 1); ctxS = contesto(caso, fam, kS, o)
            tabS, X, selez = sensibilita_ctx(caso, ctxS, o, dur_S, o['delta'])
            tabS.to_csv(os.path.join(dirc, 'S.csv' if i_l == 0 else f'S_L{kS}.csv'), index=False)
            if i_l == 0:
                riep.update(S_mesh=f"{fam} L{kS} ({int(ctxS.basis.N)} nodi, h={ctxS.h_rep * 1000:.2f} mm)", S_durata=dur_S, S_sel_tutte=','.join(selez.get('tutte', [])),
                            S_sel_purea=','.join(selez.get('purea', [])), S_sel_polpetta=','.join(selez.get('polpetta', [])))
                pd.DataFrame(X, columns=dg.PARAMETRI_CAL).to_csv(os.path.join(dirc, 'S_X.csv'), index=False)
                rum = [dict(rumore_K=s, selezione=','.join(seleziona_da_X(X, s, o['se_max']))) for s in o['rumori']]
                pd.DataFrame(rum).to_csv(os.path.join(dirc, 'S_rumore.csv'), index=False)
                if o['delta_check']:
                    tab2, X2, _ = sensibilita_ctx(caso, ctxS, o, dur_S, o['delta_check'])
                    a = tabS[tabS.sonde == 'tutte'].set_index('parametro').effetto_10pc_C; b = tab2[tab2.sonde == 'tutte'].set_index('parametro').effetto_10pc_C
                    pd.DataFrame(dict(parametro=a.index, effetto_delta=a.values, effetto_delta_check=b.reindex(a.index).values,
                                      delta=o['delta'], delta_check=o['delta_check'], dif_rel_pc=100 * (a.values - b.reindex(a.index).values) / np.where(np.abs(b.reindex(a.index).values) > 1e-9, b.reindex(a.index).values, np.nan))).to_csv(os.path.join(dirc, 'S_delta.csv'), index=False)
            del ctxS
    # ---- sensitivita' globale (sottoinsieme di casi e famiglie) ----
    if o['esegui_globale'] and fam in o['globale_famiglie'] and caso['config'] in o['globale_configurazioni']:
        kG = min(o['livello_globale'], len(dfA) - 1); ctxG = contesto(caso, fam, kG, o)
        dur_G = float(min(o['durata_misure'], np.floor(t_eval / o['passo_misura']) * o['passo_misura']))
        o_G = dict(o); o_G['globale_scale'] = tuple(o['globale_scale_config'].get(caso['config'], o['globale_scale']))
        _fase('sensitivita globale: avvio')
        info_G = analisi_globale(caso, ctxG, o_G, dur_G, dirc)
        riep.update(G_mesh=f"{fam} L{kG} ({int(ctxG.basis.N)} nodi)", G_durata=dur_G, G_simulazioni=info_G['simulazioni'], G_Q2_min=info_G['Q2_min'])
        del ctxG
    riep['tempo_s'] = time.time() - t0
    dg._json_salva(riep, os.path.join(dirc, 'riepilogo.json'))
    return dg.ricalcola_caso(dirc, cid, o, T0c)


# ---------------------------------------------------------------------------------------------------------
# Sensitivita' globale: screening di Morris + indici di Sobol da metamodello polinomiale
# ---------------------------------------------------------------------------------------------------------
def theta_nominale(p):
    return dict(sig_m20=p['_sig_m20'], alfa_m=p['_alfa_m'], sig_p20=p['_sig_p20'], alfa_p=p['_alfa_p'], fk_m=1.0, fk_p=1.0, fc_m=1.0, fc_p=1.0, fh=1.0)


def theta_da_unitario(u, p, ampiezza, scala=1.0):
    """u in [0,1]^k -> parametri: theta_j = nominale_j * (1 + scala * a_j * (2 u_j - 1)) (distribuzione uniforme di semiampiezza relativa a_j)."""
    th0 = theta_nominale(p)
    return {n: th0[n] * (1.0 + scala * ampiezza[n] * (2.0 * float(u[j]) - 1.0)) for j, n in enumerate(dg.PARAMETRI_CAL)}


USCITE_GLOBALI = ('rms_sonde', 'T_media_sonde_finale', 'incremento_T_max_finale')


def valuta_uscite(ctx, caso, o, theta, dur):
    """Tre uscite scalari: rms degli incrementi di temperatura alle sonde nel tempo (coerente con DeltaT_j della sensitivita' locale), media finale degli incrementi alle sonde e incremento finale del nodo piu' caldo."""
    pp = dg.applica_param(caso['p'], theta); pp['dt'] = float(o['dt_S']); T0 = caso['p']['T_iniziale']
    serie, _ = dg.simula_diag(ctx, pp, 1e9, durata_max=dur + 2 * o['dt_S'], t_min_run=dur)
    nomi = sorted(ctx.nomi_sonde); tempi = np.arange(o['passo_misura'], dur + 1e-9, o['passo_misura'])
    M = np.vstack([np.interp(tempi, serie['t'].values, serie['T_' + n].values) for n in nomi]) - T0
    _BARRA['sims'] += 1
    if _BARRA['tot_sims']:
        _fase(f"sensitivita globale: simulazione {_BARRA['sims']}/{_BARRA['tot_sims']}")
    return dict(rms_sonde=float(np.sqrt(np.mean(M ** 2))), T_media_sonde_finale=float(M[:, -1].mean()), incremento_T_max_finale=float(np.interp(dur, serie['t'].values, serie['T_max'].values) - T0))


def morris_disegno(k, r, rng, p=4):
    """Traiettorie di Morris (Morris, 1991) su griglia a p livelli, passo Delta = p / (2 (p - 1)), partenza e ordine casuali; effetti elementari per variazione unitaria dell'ingresso normalizzato."""
    delta = p / (2.0 * (p - 1)); livelli = np.arange(p) / (p - 1); out = []
    for _ in range(r):
        x = rng.choice(livelli, size=k); pts = [x.copy()]; passi = []
        for j in rng.permutation(k):
            su, giu = x[j] + delta <= 1 + 1e-9, x[j] - delta >= -1e-9
            sgn = (1 if rng.random() < 0.5 else -1) if (su and giu) else (1 if su else -1)
            x = x.copy(); x[j] += sgn * delta; pts.append(x.copy()); passi.append((j, sgn * delta))
        out.append((np.array(pts), passi))
    return out, delta


def morris_analisi(caso, ctx, o, dur, scala, rng):
    k = len(dg.PARAMETRI_CAL); trs, delta = morris_disegno(k, o['morris_r'], rng, o['morris_p']); ee = {u: [[] for _ in range(k)] for u in USCITE_GLOBALI}; camp = []
    for it, (pts, passi) in enumerate(trs):
        Y = [valuta_uscite(ctx, caso, o, theta_da_unitario(x, caso['p'], o['globale_ampiezza'], scala), dur) for x in pts]
        for m, (j, d) in enumerate(passi):
            for u in USCITE_GLOBALI:
                ee[u][j].append((Y[m + 1][u] - Y[m][u]) / d)
            camp.append(dict(traiettoria=it, passo=m, parametro=dg.PARAMETRI_CAL[j], delta_unitario=d, **{u: Y[m + 1][u] for u in USCITE_GLOBALI}))
    righe = []
    for u in USCITE_GLOBALI:
        mus = np.array([np.mean(np.abs(ee[u][j])) for j in range(k)]); tot = mus.sum() if mus.sum() > 0 else np.nan
        for j, n in enumerate(dg.PARAMETRI_CAL):
            e = np.array(ee[u][j]); righe.append(dict(parametro=n, uscita=u, mu=e.mean(), mu_star=np.abs(e).mean(), sigma=e.std(ddof=1) if len(e) > 1 else np.nan,
                                                      mu_star_errore_std=np.abs(e).std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan, quota_mu_star_pc=100 * np.abs(e).mean() / tot, n=len(e), scala=scala))
    return pd.DataFrame(righe), pd.DataFrame(camp)


def _termini_quadratici(Z):
    n, k = Z.shape
    cols = [np.ones(n)] + [Z[:, j] for j in range(k)] + [Z[:, j] ** 2 for j in range(k)] + [Z[:, i] * Z[:, j] for i in range(k) for j in range(i + 1, k)]
    return np.column_stack(cols)


def _fit_quadratico(Z, y, lam=1e-9):
    X = _termini_quadratici(Z); A = X.T @ X + lam * np.eye(X.shape[1]); beta = np.linalg.solve(A, X.T @ y)
    H = X @ np.linalg.solve(A, X.T); res = y - X @ beta; loo = res / (1.0 - np.diag(H)); var = np.sum((y - y.mean()) ** 2)
    return beta, dict(R2=1 - np.sum(res ** 2) / var if var > 0 else np.nan, Q2=1 - np.sum(loo ** 2) / var if var > 0 else np.nan)


def _sobol_analitici(beta, k):
    """Indici di Sobol ESATTI del metamodello quadratico con ingressi indipendenti e uniformi in [-1, 1] (nessun rumore di campionamento).
    Termini ortogonali: a_j z_j (varianza a_j^2/3), b_j z_j^2 (b_j^2 * 4/45), c_ij z_i z_j (c_ij^2/9). S_j = V_j/V; S_Tj = (V_j + somma delle interazioni che contengono j)/V."""
    a_, b_, c_ = beta[1:1 + k], beta[1 + k:1 + 2 * k], beta[1 + 2 * k:]
    coppie = [(i, j) for i in range(k) for j in range(i + 1, k)]
    Vj = a_ ** 2 / 3.0 + b_ ** 2 * 4.0 / 45.0; Vij = c_ ** 2 / 9.0; V = Vj.sum() + Vij.sum(); inter = np.zeros(k)
    for (i, j), v in zip(coppie, Vij):
        inter[i] += v; inter[j] += v
    if V <= 0:
        return np.full(k, np.nan), np.full(k, np.nan)
    return Vj / V, (Vj + inter) / V


def sobol_metamodello(caso, ctx, o, dur, scala, rng):
    """Campione ipercubo latino (scrambled) di lhs_n simulazioni -> metamodello polinomiale di grado 2 (con interazioni) -> indici di Sobol con intervalli bootstrap.
    Metamodello e indici: Sudret (2008); campione ipercubo latino: McKay et al. (1979); intervalli bootstrap: Dubreuil et al. (2014). La validita' e' misurata da Q2 (leave-one-out): se Q2 < 0.9 gli indici NON sono affidabili (si segnala in G_qualita.csv)."""
    from scipy.stats import qmc
    k = len(dg.PARAMETRI_CAL); U = qmc.LatinHypercube(d=k, seed=int(rng.integers(1 << 31))).random(o['lhs_n']); Z = 2 * U - 1
    Y = [valuta_uscite(ctx, caso, o, theta_da_unitario(u, caso['p'], o['globale_ampiezza'], scala), dur) for u in U]
    camp = pd.DataFrame(U, columns=['u_' + n for n in dg.PARAMETRI_CAL]); righe, qual = [], []
    for u in USCITE_GLOBALI:
        y = np.array([r[u] for r in Y]); camp[u] = y; beta, q = _fit_quadratico(Z, y)
        S1, ST = _sobol_analitici(beta, k); b1, bT = [], []
        for _ in range(o['sobol_boot']):
            idx = rng.integers(0, len(y), len(y))
            try:
                bb, _ = _fit_quadratico(Z[idx], y[idx]); s1, st = _sobol_analitici(bb, k); b1.append(s1); bT.append(st)
            except np.linalg.LinAlgError:
                continue
        b1, bT = np.array(b1), np.array(bT)
        for j, n in enumerate(dg.PARAMETRI_CAL):
            righe.append(dict(parametro=n, uscita=u, S1=S1[j], S1_inf=np.nanpercentile(b1[:, j], 2.5) if len(b1) else np.nan, S1_sup=np.nanpercentile(b1[:, j], 97.5) if len(b1) else np.nan,
                              ST=ST[j], ST_inf=np.nanpercentile(bT[:, j], 2.5) if len(bT) else np.nan, ST_sup=np.nanpercentile(bT[:, j], 97.5) if len(bT) else np.nan, scala=scala))
        qual.append(dict(uscita=u, R2=q['R2'], Q2=q['Q2'], n_addestramento=len(y), n_termini=_termini_quadratici(Z[:2]).shape[1], scala=scala, affidabile=bool(q['Q2'] >= 0.9)))
    return pd.DataFrame(righe), pd.DataFrame(qual), camp


def analisi_globale(caso, ctx, o, dur, dirc):
    """Sensitivita' globale di un caso: Morris + Sobol da metamodello, per ogni scala degli intervalli. Scrive G_morris*.csv, G_sobol*.csv, G_qualita*.csv, G_campioni*.csv."""
    rng = np.random.default_rng(int(o['globale_seed']) + (zlib.crc32(caso['id'].encode()) % 100000)); nsim = 0; q2 = []
    _BARRA['sims'] = 0; _BARRA['tot_sims'] = len(o['globale_scale']) * (o['morris_r'] * (len(dg.PARAMETRI_CAL) + 1) + o['lhs_n'])
    for scala in o['globale_scale']:
        suf = '' if scala == 1.0 else f'_x{scala:g}'
        M, Mc = morris_analisi(caso, ctx, o, dur, scala, rng); M.to_csv(os.path.join(dirc, f'G_morris{suf}.csv'), index=False); Mc.to_csv(os.path.join(dirc, f'G_morris_campioni{suf}.csv'), index=False)
        Sb, Q, camp = sobol_metamodello(caso, ctx, o, dur, scala, rng); Sb.to_csv(os.path.join(dirc, f'G_sobol{suf}.csv'), index=False); Q.to_csv(os.path.join(dirc, f'G_qualita{suf}.csv'), index=False); camp.to_csv(os.path.join(dirc, f'G_campioni{suf}.csv'), index=False)
        nsim += o['morris_r'] * (len(dg.PARAMETRI_CAL) + 1) + o['lhs_n']; q2 += list(Q.Q2.values)
    _BARRA['tot_sims'] = 0
    return dict(simulazioni=nsim, Q2_min=float(np.nanmin(q2)))


def tabella_fonti_intervalli():
    """Fonti degli intervalli dei parametri per la sensitivita' globale. Regola: distribuzione uniforme con semiampiezza relativa a = sqrt(3) x scarto tipo relativo di letteratura
    (l'uniforme di semiampiezza a ha deviazione standard a/sqrt(3)). 'scala 2' raddoppia la semiampiezza (verifica della dipendenza dalla scelta degli intervalli).
    Affidabilita': 'solida' = studio primario con valore numerico verificato; 'debole' = tesi o dato indiretto: sostituire con l'incertezza di calibrazione o con una fonte primaria."""
    R = [
        ("sig_m20, sig_p20", 0.10, "~10% (massimo)",
         "Deviazioni del modello a circuito per la conducibilita' efficace di miscele a bassa acidita' inferiori al 10%; differenze fra conducibilita' misurata e di riferimento in cella ohmica 0.87-6.75% (tesi, Chulalongkorn University, repository ETD). La dipendenza lineare dalla temperatura e' confermata da Palaniappan e Sastry (1991) e Sarang, Sastry e Knipe (2008).",
         "intervallo assunto; le conducibilita' delle ricette derivano da misure (grafico e tesi): l'incertezza di misura andra' sostituita a quella assunta"),
        ("alfa_m, alfa_p", 0.41, "24% (scarto tipo relativo: -1.42 +- 0.34 %/C; intervallo -0.58/-2.27 %/C)",
         "Coefficiente relativo di temperatura dell'impedenza di tessuti animali e vegetali, 20-50 C; Journal of Electrical Bioimpedance (2024), DOI 10.2478/joeb-2024-0013 (autori da completare). Per la carne scaldata rapidamente (oltre 20 C/min) la conducibilita' e' lineare nella temperatura (modello del Wageningen, J. Food Eng. 208, 2017, 37-47).",
         "solida per l'ordine di grandezza; vale per l'impedenza, non specificamente per la purea"),
        ("fk_m, fk_p", 0.12, "7% (errore percentuale standard medio)",
         "Modelli basati su composizione e temperatura predicono la conducibilita' termica della carne bovina in cottura con errore standard percentuale medio di circa 7% (Shahedi Baghe-Khandan e Okos, J. Food Sci. 46, 1981, 1301-1305: cifra da verificare sul testo); ripetibilita' misurata 3.2-5.2% (carne di cammello, J. King Saud Univ. Agric. Sci. 16(2), 2004, 153-160).",
         "solida per la carne; non per la purea"),
        ("fc_m, fc_p", 0.08, "4.7% (errore massimo del modello)",
         "Proprieta' termiche dai modelli per composizione entro 3.9% dai valori di letteratura di alimenti liquidi ed entro 4.7% dai valori sperimentali di latte evaporato, succo d'arancia e salsiccia bratwurst (Choi, dissertazione Purdue, relatore Okos). Per alimenti porosi gli errori possono essere molto maggiori (Carson, ICFP 2018).",
         "solida; l'incertezza sulla composizione (acqua) non e' inclusa"),
        ("fh", 0.35, "20% (RMSRE della correlazione di Churchill e Chu)",
         "Scarto quadratico medio relativo di circa 20% della correlazione di Churchill e Chu (1975) per cilindri orizzontali sui dati originali (Jaffer e Jaffer, arXiv:2512.06019); l'irraggiamento pesa fino al 60% del calore scambiato nei dati piu' vecchi. Include solo l'incertezza della correlazione, non quella di emissivita' ed evaporazione.",
         "solida per la correlazione; il caso reale (cella con superficie libera) e' diverso"),
    ]
    return pd.DataFrame(R, columns=['parametri', 'semiampiezza_relativa', 'scarto_tipo_di_letteratura', 'fonte_e_nota', 'affidabilita_e_limite'])


def riferimenti_metodi_globali():
    """Riferimenti verificati per la sensitivita' globale."""
    return [
        "Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. Technometrics 33, 161-174. DOI 10.1080/00401706.1991.10484804.",
        "McKay, M.D., Beckman, R.J., Conover, W.J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics 21, 239-245.",
        "Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions. Reliability Engineering and System Safety 93(7), 964-979.",
        "Dubreuil, S., Berveiller, M., Petitjean, F., Salaun, M. (2014). Construction of bootstrap confidence intervals on sensitivity indices computed by polynomial chaos expansion. Reliability Engineering and System Safety 121, 263-275.",
        "Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., et al. (2008). Global Sensitivity Analysis: The Primer. Wiley.",
    ]


def riepilogo_globale(base):
    """Aggrega la sensitivita' globale: sintesi Morris e Sobol per famiglia, qualita' del metamodello e confronto con la sensitivita' locale (correlazione di rango e sovrapposizione dei primi 4 parametri)."""
    from scipy.stats import spearmanr
    Mo, So, Qu, Cf = [], [], [], []
    for fam in FAMIGLIE:
        d0 = os.path.join(base, fam, 'casi')
        if not os.path.isdir(d0):
            continue
        for cid in sorted(os.listdir(d0)):
            d = os.path.join(d0, cid); fm = os.path.join(d, 'G_morris.csv')
            if not os.path.exists(fm):
                continue
            m = pd.read_csv(fm); s = pd.read_csv(os.path.join(d, 'G_sobol.csv')); q = pd.read_csv(os.path.join(d, 'G_qualita.csv'))
            for x, L in ((m, Mo), (s, So), (q, Qu)):
                x = x.copy(); x['famiglia'] = fam; x['caso'] = cid; L.append(x)
            fl = os.path.join(d, 'S.csv')
            if os.path.exists(fl):
                loc = pd.read_csv(fl); loc = loc[loc.sonde == 'tutte'].set_index('parametro').effetto_10pc_C.reindex(dg.PARAMETRI_CAL)
                mm = m[m.uscita == 'rms_sonde'].set_index('parametro').mu_star.reindex(dg.PARAMETRI_CAL); ss = s[s.uscita == 'rms_sonde'].set_index('parametro').ST.reindex(dg.PARAMETRI_CAL)
                top = lambda v: set(v.sort_values(ascending=False).index[:4])
                Cf.append(dict(famiglia=fam, caso=cid, rho_locale_morris=spearmanr(loc, mm)[0], rho_locale_sobolST=spearmanr(loc, ss)[0],
                               top4_comuni_locale_morris=len(top(loc) & top(mm)), top4_comuni_locale_sobolST=len(top(loc) & top(ss))))
    out = {}
    if Mo:
        Mo, So, Qu = pd.concat(Mo, ignore_index=True), pd.concat(So, ignore_index=True), pd.concat(Qu, ignore_index=True); out.update(morris=Mo, sobol=So, qualita=Qu)
        rat = Mo.assign(rapporto=Mo.sigma / Mo.mu_star.replace(0, np.nan))
        out['morris_sintesi'] = rat.groupby(['famiglia', 'uscita', 'parametro']).agg(mu_star_mediano=('mu_star', 'median'), quota_pc_mediana=('quota_mu_star_pc', 'median'), sigma_su_mu_star_mediano=('rapporto', 'median')).reset_index()
        out['sobol_sintesi'] = So.groupby(['famiglia', 'uscita', 'parametro']).agg(S1_mediano=('S1', 'median'), ST_mediano=('ST', 'median'), ST_inf_mediano=('ST_inf', 'median'), ST_sup_mediano=('ST_sup', 'median'),
                                                                                 casi_ST_inf_oltre_5pc=('ST_inf', lambda x: 100 * np.mean(x > 0.05))).reset_index()
        out['qualita_sintesi'] = Qu.groupby(['famiglia', 'uscita']).agg(Q2_minimo=('Q2', 'min'), Q2_mediano=('Q2', 'median'), casi=('caso', 'nunique'), affidabili_pc=('affidabile', lambda x: 100 * np.mean(x))).reset_index()
    if Cf:
        out['confronto_locale_globale'] = pd.DataFrame(Cf)
    Cs = []
    for fam in FAMIGLIE:
        d0 = os.path.join(base, fam, 'casi')
        if not os.path.isdir(d0):
            continue
        for cid in sorted(os.listdir(d0)):
            f1, f2 = os.path.join(d0, cid, 'G_sobol.csv'), os.path.join(d0, cid, 'G_sobol_x2.csv')
            if os.path.exists(f1) and os.path.exists(f2):
                a1 = pd.read_csv(f1).query("uscita == 'rms_sonde'").set_index('parametro').ST.reindex(dg.PARAMETRI_CAL); a2 = pd.read_csv(f2).query("uscita == 'rms_sonde'").set_index('parametro').ST.reindex(dg.PARAMETRI_CAL)
                t4 = lambda v: set(v.sort_values(ascending=False).index[:4])
                Cs.append(dict(famiglia=fam, caso=cid, rho_scala1_scala2=spearmanr(a1, a2)[0], top4_comuni=len(t4(a1) & t4(a2)), differenza_ST_massima=float((a1 - a2).abs().max())))
    if Cs:
        out['confronto_scale_intervalli'] = pd.DataFrame(Cs)
    return out



def _analizza_str(base):
    return lambda caso, cartella, opz: analizza_famiglia(caso, 'strutturata', cartella, opz, base)


def _analizza_conf(base):
    return lambda caso, cartella, opz: analizza_famiglia(caso, 'conforme', cartella, opz, base)


def _copia_pilota(src, dst):
    """Copia i file del tempo di valutazione condiviso (piccoli json) fra locale e Drive, solo quelli mancanti nella destinazione."""
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True); n = 0
    for f in os.listdir(src):
        if not os.path.exists(os.path.join(dst, f)):
            try:
                import shutil
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f)); n += 1
            except OSError:
                pass
    return n


def esegui_articolo(ricette_sel, config_sel, cartella='risultati_articolo', opz=None, campo_V_cm=4.0, T0=20.0, T_amb=21.0, riprendi=True, verbose=True,
                    cartella_drive=None, famiglie=FAMIGLIE, barra=True, casi=None):
    """Ciclo su tutti i casi: per ogni caso entrambe le famiglie (cosi' i risultati parziali sono sempre confrontabili).
    Barra di avanzamento (tqdm): una unita' = un caso in una famiglia; mostra tempo trascorso, tempo stimato, caso, famiglia e fase in corso.
    Checkpoint: ogni caso e' salvato in locale e copiato sul Drive a calcolo finito; dopo un riavvio i casi completati si ripristinano dal Drive e si saltano
    (un caso e' completo solo se esistono riepilogo.json e A.csv, scritti alla fine). Un'interruzione a meta' caso fa rifare solo quel caso."""
    o = opzioni_art(opz)
    base = cartella
    os.makedirs(base, exist_ok=True)
    if cartella_drive:                                       # ripristino del tempo di valutazione condiviso e dei casi gia' completati
        n_p = _copia_pilota(os.path.join(cartella_drive, 'pilota'), os.path.join(base, 'pilota'))
        if verbose and n_p:
            print(f"Ripristinati dal Drive {n_p} tempi di valutazione condivisi")
        for fam in famiglie:
            dg.ripristina_da_drive(os.path.join(base, fam), os.path.join(cartella_drive, fam), verbose)
    coppie = list(casi) if casi else [(r, c) for r in ricette_sel for c in config_sel]            # casi: elenco di coppie (ricetta, configurazione), p.es. il DoE
    previsti = [f'{r}__{c}' for r, c in coppie]
    gia = sum(os.path.exists(os.path.join(base, fam, 'casi', cid, 'riepilogo.json')) for fam in famiglie for cid in previsti) if riprendi else 0
    bar = _tqdm(total=len(previsti) * len(famiglie), initial=gia, unit='caso', dynamic_ncols=True, smoothing=0.1) if (barra and _tqdm is not None) else None
    _BARRA['bar'] = bar
    scrivi = (lambda m: bar.write(m)) if bar is not None else print
    log = []
    try:
        for r, c in coppie:
            if True:
                cid = f'{r}__{c}'
                for fam in famiglie:
                    fn = _analizza_str(base) if fam == 'strutturata' else _analizza_conf(base)
                    cd = os.path.join(cartella_drive, fam) if cartella_drive else None
                    f_ok = os.path.join(base, fam, 'casi', cid, 'riepilogo.json')
                    fatto = riprendi and os.path.exists(f_ok)
                    if bar is not None and not fatto:
                        bar.set_description(f'{cid} [{fam}]', refresh=False); _fase('avvio')
                    errori = dg.esegui_tutti([r], [c], os.path.join(base, fam), o, campo_V_cm, T0, T_amb, riprendi, False, cd, analizza=fn)
                    log += errori
                    if not fatto:
                        if errori:
                            scrivi(f"[ERRORE] {cid} [{fam}]: {errori[-1].get('errore', '')}")
                        elif verbose and os.path.exists(f_ok):
                            ri = json.load(open(f_ok)); scrivi(f"{cid} [{fam}]: ok in {ri.get('tempo_s', float('nan')):.0f} s | t_val = {ri.get('t_valutazione')} s")
                    if bar is not None:
                        bar.update(1)
                    dg.scrivi_checkpoint(os.path.join(base, fam), o, previsti, extra=dict(famiglia=fam, protocollo='articolo'))      # checkpoint con TUTTI i casi previsti
                    if cartella_drive:
                        dg.sincronizza_su_drive(os.path.join(base, fam), os.path.join(cartella_drive, fam), casi=[])
                        _copia_pilota(os.path.join(base, 'pilota'), os.path.join(cartella_drive, 'pilota'))
    finally:
        if bar is not None:
            bar.close()
        _BARRA['bar'] = None; _BARRA['tot_sims'] = 0
    return log


# ---------------------------------------------------------------------------------------------------------
# Verifica del codice con soluzioni analitiche
# ---------------------------------------------------------------------------------------------------------
def _caso_RP(cella=dg.CELLA_UNISA):
    return dg.costruisci_caso('A_40V', 'RP')


def verifica_adiabatica_analitica(o=None, dts=(40, 20, 10, 5, 2.5), t_eval=270.0, sigma20=1.0, alfa=0.03, rhocp=3.5e6, campo_V_cm=4.0):
    """Riscaldamento uniforme adiabatico di una sola fase con sigma(T) lineare e proprieta' costanti: soluzione esatta
         T(t) = T0 + [exp(alfa*sigma20*E^2*t/(rho*cp)) - 1]/alfa ,   E = V_rms/Lx .
    Verifica (senza errore di mesh) lo schema temporale e l'accoppiamento elettro-termico sfalsato: ordine atteso 1."""
    o = opzioni_art(o); righe = []
    for fam in FAMIGLIE:
        caso = _caso_RP(); p = caso['p']
        p.update(prop_modo='manuali', k_matrice=0.5, rho_matrice=1000.0, cp_matrice=rhocp / 1000.0, h_modo='costante', fattore_h=0.0, matrice_a=sigma20 * alfa, matrice_b=sigma20 * (1 - 20 * alfa),
                 V_rms=campo_V_cm * 100.0 * caso['geo']['Lx'], T_iniziale=20.0, T_amb=20.0, fk_matrice=1.0, fc_matrice=1.0)
        E = p['V_rms'] / caso['geo']['Lx']
        T_ex = 20.0 + (np.exp(alfa * sigma20 * E ** 2 * t_eval / rhocp) - 1.0) / alfa
        oo = dict(o); oo.update(nodi_grossa=300, livelli=3, max_nodi_conforme=20000)
        ctx = contesto(caso, fam, 0, oo)
        prev = None
        for d_ in dts:
            pp = dict(p); pp['dt'] = float(d_)
            serie, _ = dg.simula_diag(ctx, pp, 1e9, durata_max=t_eval + 2 * d_, t_min_run=t_eval)
            Tn = float(np.interp(t_eval, serie['t'].values, serie['T_max'].values)); err = Tn - T_ex
            righe.append(dict(famiglia=fam, dt=d_, T_numerica=Tn, T_esatta=T_ex, errore=err, errore_rel_incremento_pc=100 * err / (T_ex - 20.0),
                              ordine_osservato=(np.log(abs(prev / err)) / np.log(2.0) if prev is not None and err != 0 else np.nan), chiusura_energia=float(serie.chiusura_energia.abs().max()),
                              n_nodi=int(ctx.basis.N)))
            prev = err
    return pd.DataFrame(righe)


def verifica_sfera_campo_uniforme(o=None, contrasti=(0.2, 4.4), L_mm=60.0, D_mm=20.0, h_mm=(4.0, 3.0, 2.2, 1.6, 1.2)):
    """Sfera isolata in un campo uniforme (cubo con pareti laterali isolanti, elettrodi alle facce x = 0 e x = L). Per una sfera in mezzo infinito il campo interno e'
    uniforme e vale E_in = 3 s_m / (s_p + 2 s_m) E0: la sorgente media nella sfera e' q_p = V^2 s_p (3 s_m/(s_p + 2 s_m))^2 / L^2. La dimensione h e' quella degli
    elementi all'interfaccia per entrambe le famiglie (conforme: dimensione globale 2h). Il confinamento (R/L = 0.167) lascia un errore residuo che NON svanisce con
    il raffinamento: si legge come scarto di plateau dal valore analitico, uguale per le due famiglie, ed e' separato dall'errore di discretizzazione."""
    o = opzioni_art(o); righe = []
    geo = me.geometria(dg.CELLA_UNISA, 'RP (solo purea)', dict(Lx=L_mm * 1e-3, Ly=L_mm * 1e-3, Lz=L_mm * 1e-3, D=D_mm * 1e-3, occupazione=[False, True, False]))
    Lx = geo['Lx']
    for c in contrasti:
        p = dict(me.PARAMETRI_DEFAULT); p.update(matrice_a=0.0, matrice_b=1.0, polpetta_a=0.0, polpetta_b=float(c), V_rms=1.0, T_iniziale=20.0, T_iniziale_polpetta=20.0)
        q_ex = 1.0 * c * (3.0 / (c + 2.0)) ** 2 / Lx ** 2
        caso = dict(id='sfera', ricetta='sfera', config='centro', cella=dg.CELLA_UNISA, geo=geo, p=p)
        for fam in FAMIGLIE:
            for k, h in enumerate(h_mm):
                if fam == 'strutturata':
                    n = tuple(max(2, int(round(L_mm / h))) for _ in range(3)); ctx = dg.Contesto(geo, n, p, costruisci_filtro=False)
                else:
                    oc = dict(o); oc.update(lc_base_mm=2.0 * h, fattore_lc=1.0, livelli_conf=1, rapporto_lci=0.5, cartella_mesh=os.path.join(o['cartella_mesh'], 'sfera'))
                    ctx = dgc.contesto_conforme(dict(caso), 0, oc)
                e = dg.solo_elettrico(ctx, p); err = 100 * (e['q_media_polp'] - q_ex) / q_ex
                righe.append(dict(famiglia=fam, contrasto=c, h_interfaccia_mm=h, h_rep_mm=ctx.h_rep * 1000, n_nodi=int(ctx.basis.N), R_su_h=(D_mm / 2) / h, q_numerica=e['q_media_polp'], q_analitica=q_ex,
                                  errore_pc=err, errore_volume_sfera_pc=100 * e['errore_volume'], ordine_osservato=np.nan))
                del ctx
    D = pd.DataFrame(righe)
    for (fam, c), g in D.groupby(['famiglia', 'contrasto']):
        idx = g.index.tolist()
        for a_, b_ in zip(idx[:-1], idx[1:]):
            ea, eb = abs(D.loc[a_, 'errore_pc']), abs(D.loc[b_, 'errore_pc']); ha, hb = D.loc[a_, 'h_interfaccia_mm'], D.loc[b_, 'h_interfaccia_mm']
            if ea > 0 and eb > 0 and ha != hb:
                D.loc[b_, 'ordine_osservato'] = np.log(ea / eb) / np.log(ha / hb)
    return D


# ---------------------------------------------------------------------------------------------------------
# Validazione degli stimatori (leave-finest-out) e analisi dei test di robustezza
# ---------------------------------------------------------------------------------------------------------
def validazione_stimatori(cartella_fam, opz=None):
    """Per ogni caso e grandezza: stima con le mesh piu' grossolane (tutte tranne la piu' fine) e confronto con la mesh piu' fine.
    Errore di riferimento della mesh piu' fine = |f(penultima) - f(piu' fine)| / (r^p - 1) (estrapolazione di Richardson), con r = rapporto fra le h delle due mesh piu' fini
    e p osservato sulle tre mesh piu' fini (p = 1 se non monotona)."""
    base = os.path.join(cartella_fam, 'casi'); righe = []
    for cid in sorted(os.listdir(base)):
        d = os.path.join(base, cid)
        if not os.path.exists(os.path.join(d, 'riepilogo.json')):
            continue
        A = pd.read_csv(os.path.join(d, 'A.csv')); B = pd.read_csv(os.path.join(d, 'B_tempo_fisso.csv'))
        T0 = 20.0; rise = float(B['T_max'].values[np.argmin(B['h_rep_mm'].values)] - T0)
        for liv, studio, col, etich in dg.QOI_ARTICOLO:
            D = A if studio == 'A' else B
            if col not in D.columns or len(D) < 5:
                continue
            D = D.sort_values('h_rep_mm'); h = D.h_rep_mm.values.astype(float); f = D[col].values.astype(float)       # ascendente: f[0] = mesh piu' fine
            eg = dg.incertezza_discretizzazione(h[1:], f[1:], 'gci3'); el = dg.incertezza_discretizzazione(h[1:], f[1:], 'ls')
            g3 = dg.gci_celik(h[:3], f[:3]); pr = float(np.clip(g3['p_oss'], 0.5, 2.0)) if (g3['regime'] == 'monotona' and np.isfinite(g3['p_oss'])) else 1.0
            r_raff = float(h[1] / h[0]); e_vero = abs(f[1] - f[0]) / (r_raff ** pr - 1.0); scala = rise if col in dg._T_QOI else abs(f[0])
            righe.append(dict(caso=cid, livello=liv, grandezza=col, e_vero=e_vero, scala=scala, U_GCI3=eg['U'], U_LS=el['U'], regime=eg['regime'],
                              rapporto_GCI3=eg['U'] / e_vero if e_vero > 0 else np.nan, rapporto_LS=el['U'] / e_vero if e_vero > 0 else np.nan,
                              copre_GCI3=bool(eg['U'] >= e_vero), copre_LS=bool(el['U'] >= e_vero)))
    if not righe:
        return pd.DataFrame(), pd.DataFrame()                 # meno di 5 livelli: validazione non eseguibile
    V = pd.DataFrame(righe)
    S = []
    for lv, g in list(V.groupby('livello')) + [('tutti', V)]:
        for nome, cc, rr in (('GCI (3 mesh)', 'copre_GCI3', 'rapporto_GCI3'), ('Minimi quadrati', 'copre_LS', 'rapporto_LS')):
            S.append(dict(livello=lv, stimatore=nome, grandezze=len(g), copertura_pc=100 * g[cc].mean(), rapporto_mediano=float(np.nanmedian(g[rr])),
                          rapporto_p10=float(np.nanpercentile(g[rr], 10)), rapporto_p90=float(np.nanpercentile(g[rr], 90))))
    return V, pd.DataFrame(S)


def analisi_errore_iterativo(cartella_fam, T_num_rel):
    """Scarto massimo (in % della scala) di ogni grandezza fra le tolleranze testate e la piu' stretta, e rapporto con l'incertezza numerica U_num.
    T_num_rel: DataFrame con colonne caso, grandezza, U_num_rel_pc (da tabella_incertezza_tutti)."""
    base = os.path.join(cartella_fam, 'casi'); righe = []
    for cid in sorted(os.listdir(base)):
        f = os.path.join(base, cid, 'iter.csv')
        if not os.path.exists(f):
            continue
        D = pd.read_csv(f).sort_values('rtol'); ref = D.iloc[0]; rise = float(ref['T_max'] - 20.0)
        for liv, studio, col, etich in dg.QOI_ARTICOLO:
            if col not in D.columns:
                continue
            scala = rise if col in dg._T_QOI else abs(ref[col])
            for _, r in D.iloc[1:].iterrows():
                righe.append(dict(caso=cid, grandezza=col, rtol=r['rtol'], scarto_pc=100 * abs(r[col] - ref[col]) / scala if scala else np.nan, solutore_el=r['solutore_el']))
    I = pd.DataFrame(righe)
    if I.empty:
        return I
    u = T_num_rel[['caso', 'grandezza', 'U_num_rel_pc']]
    I = I.merge(u, on=['caso', 'grandezza'], how='left'); I['rapporto_scarto_U'] = I.scarto_pc / I.U_num_rel_pc
    return I


def confronto_famiglie(base, o=None, T0=20.0):
    """Differenza fra i limiti GCI delle due famiglie per tutti i casi (indicatore di non-asintoticita' della famiglia strutturata, NON errore di modello: con rampa ~ h la rappresentazione diffusa tende a quella netta) e confronto delle sensitivita' a parita' di mesh e durata."""
    o = opzioni_art(o)
    Ts = dg.tabella_incertezza_tutti(os.path.join(base, 'strutturata'), o, T0, 'gci3'); Tc = dg.tabella_incertezza_tutti(os.path.join(base, 'conforme'), o, T0, 'gci3')
    m = Ts.merge(Tc, on=['caso', 'grandezza'], suffixes=('_s', '_c'))
    m['E_int'] = m.valore_limite_s - m.valore_limite_c; m['E_rel_pc'] = 100 * m.E_int / m.scala_c; m['U_int'] = m.U_h_s + m.U_h_c; m['oltre_incertezza'] = m.E_int.abs() > m.U_int
    # sensitivita'
    righe = []
    for cid in sorted(os.listdir(os.path.join(base, 'conforme', 'casi'))):
        fs, fc = os.path.join(base, 'strutturata', 'casi', cid, 'S.csv'), os.path.join(base, 'conforme', 'casi', cid, 'S.csv')
        if os.path.exists(fs) and os.path.exists(fc):
            a = pd.read_csv(fs); b = pd.read_csv(fc); a = a[a.sonde == 'tutte'].set_index('parametro'); b = b[b.sonde == 'tutte'].set_index('parametro')
            for p_ in a.index:
                righe.append(dict(caso=cid, parametro=p_, effetto_s=a.loc[p_, 'effetto_10pc_C'], effetto_c=b.loc[p_, 'effetto_10pc_C'], sel_s=bool(a.loc[p_, 'selezionato']), sel_c=bool(b.loc[p_, 'selezionato'])))
    S = pd.DataFrame(righe)
    if not S.empty:
        S['dif_rel_pc'] = 100 * (S.effetto_s - S.effetto_c) / S.effetto_c.replace(0, np.nan); S['stessa_selezione'] = S.sel_s == S.sel_c
    return m, S


# ---------------------------------------------------------------------------------------------------------
# Riproducibilita'
# ---------------------------------------------------------------------------------------------------------
def riproducibilita(cartella, opz=None, file_moduli=('multifase_engine.py', 'diagnostica_ohmico.py', 'diagnostica_conforme.py', 'protocollo_articolo.py')):
    """Salva ambiente, versioni dei pacchetti, impronte SHA-256 dei file e opzioni in cartella/riproducibilita.json."""
    ver = {}
    for nome in ('numpy', 'scipy', 'pandas', 'skfem', 'matplotlib', 'pyamg', 'gmsh', 'meshio', 'openpyxl', 'docx'):
        try:
            mod = __import__(nome); ver[nome] = getattr(mod, '__version__', 'n.d.')
        except Exception:
            ver[nome] = 'non installato'
    imp = {}
    for f in file_moduli:
        if os.path.exists(f):
            imp[f] = hashlib.sha256(open(f, 'rb').read()).hexdigest()
    d = dict(data=datetime.datetime.now().isoformat(timespec='seconds'), python=sys.version, piattaforma=platform.platform(), processore=platform.processor(), pacchetti=ver, sha256=imp,
             opzioni={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in opzioni_art(opz).items()},
             tolleranze=dict(rtol_elettrico=me.RTOL_ELETTRICO, rtol_termico=me.RTOL_TERMICO))
    os.makedirs(cartella, exist_ok=True)
    json.dump(d, open(os.path.join(cartella, 'riproducibilita.json'), 'w'), indent=1)
    return d


def tabella_critiche():
    """Matrice critiche dei revisori -> risposta nell'analisi -> dove si trova l'evidenza -> limite residuo (da dichiarare nell'articolo)."""
    R = [
        ("Il codice e' corretto? (verifica del codice)", "Soluzione analitica del riscaldamento adiabatico uniforme (schema temporale e accoppiamento) e della sfera in campo uniforme (conducibilita' e interfaccia), su entrambe le discretizzazioni; chiusura dell'energia, P = V I e sbilancio delle correnti a ogni caso", "verifica_adiabatica.csv, verifica_sfera.csv, riepilogo (controlli)", "il test della sfera ha un errore di confinamento (R/L)^3 dichiarato; non e' una verifica con soluzioni manufatte dell'intero sistema accoppiato non lineare"),
        ("Le mesh sono troppo grossolane / poche griglie per stimare l'incertezza", "5 livelli per famiglia con rapporto sqrt(2) in h e stesso numero di nodi per le due famiglie; GCI su 3 mesh e minimi quadrati su tutte; controllo del regime asintotico e conteggio dei casi non monotoni", "A.csv, B_tempo_fisso.csv, tabelle di incertezza", "le stime non sono limiti garantiti (Eca et al., 2021)"),
        ("Gli stimatori di incertezza sono affidabili?", "Validazione leave-finest-out: stima con le mesh grossolane e confronto con la piu' fine (copertura e rapporto incertezza/errore) per entrambi gli stimatori e le due famiglie", "validazione_stimatori_*.csv", "l'errore 'vero' e' estrapolato dalla mesh piu' fine, non esatto"),
        ("L'errore iterativo e' trascurabile?", "Rieseguito a tolleranze 1e-4, 1e-6, 1e-8, 1e-12: scarto rispetto alla piu' stretta e rapporto con U_num", "iter.csv, errore_iterativo_*.csv", "per mesh sotto 3000 nodi il solutore elettrico e' diretto (nessun errore iterativo elettrico)"),
        ("L'errore temporale dipende dalla mesh?", "Studio del passo su due mesh (secondo e terzo livello) e confronto di U_dt", "C.csv, C_mesh2.csv", "solo due mesh"),
        ("Come si combinano le componenti?", "Somma aritmetica U_h + U_dt (conservativa; Eca e Hoekstra, 2009); estensione a due componenti di discretizzazione dichiarata", "protocollo (documento sul criterio)", "scelta del protocollo, non una regola di letteratura"),
        ("La rappresentazione diffusa dell'interfaccia introduce un errore di modello?", "Con rampa di larghezza proporzionale a h la rappresentazione diffusa tende a quella netta per h -> 0: verificato con la sfera in campo uniforme (soluzione analitica), dove entrambe le famiglie convergono al valore esatto con ordine circa 1. La differenza fra i limiti stimati delle due famiglie (confronto_interfaccia.csv) NON e' un errore di modello: misura l'incertezza di estrapolazione della famiglia strutturata, che non e' in regime asintotico", "verifica_sfera.csv, confronto_interfaccia.csv", "l'errore di discretizzazione all'interfaccia e' ordine 1 (piu' lento dell'ordine 2 atteso): va dichiarato; la mesh conforme approssima la sfera con facce piane (errore di volume riportato)"),
        ("I due tipi di mesh sono confrontabili?", "Famiglie armonizzate (stesso numero di nodi a ogni livello, h = (V/N)^(1/3) con la stessa definizione), tempo di valutazione condiviso, stesso livello e durata per la sensitivita'", "opzioni in riproducibilita.json", "le due famiglie restano discretizzazioni diverse: il confronto riguarda le stime, non un'unica soluzione esatta"),
        ("La sensitivita' dipende dal passo delta delle differenze finite?", "Doppio delta (0.10 e 0.05) e confronto degli effetti", "S_delta.csv", "sensitivita' locale"),
        ("La selezione dei parametri dipende dal rumore assunto?", "Selezione ripetuta per rumore 0.1, 0.25, 0.5, 1.0 K", "S_rumore.csv", "il rumore reale dei sensori resta da fornire"),
        ("La sensitivita' dipende dalla mesh?", "Sensitivita' su due mesh (casi approfonditi) e confronto fra famiglie a parita' di livello e durata", "S_L*.csv, confronto_sensitivita.csv", "due mesh nei soli casi approfonditi"),
        ("Sensitivita' locale vs globale", "Analisi globale su tutti i casi (mesh conforme): screening di Morris (Morris, 1991) e indici di Sobol (S_i e S_Ti con intervalli bootstrap) da metamodello polinomiale validato con Q2 leave-one-out (Sudret, 2008; Dubreuil et al., 2014); intervalli dei parametri ricavati dalla letteratura (uniforme con deviazione standard di letteratura) e raddoppiati nella configurazione Tris per verificare la dipendenza dalla scelta; confronto di rango con la sensitivita' locale", "G_morris*.csv, G_sobol*.csv, G_qualita*.csv, fonti_intervalli_globali.csv", "intervalli di sigma20 poggiati su fonti deboli (sostituire con l'incertezza di calibrazione); Sobol su metamodello (Q2 dichiarato); solo mesh conforme di default"),
        ("Risultati oltre il campo di validita' (ebollizione)", "Tempo di valutazione limitato a T_max = 90 C, condiviso fra le famiglie; sensitivita' sulla stessa durata", "riepilogo (t_valutazione)", "il modello non include l'ebollizione"),
        ("Riproducibilita'", "Versioni dei pacchetti, impronte SHA-256 dei file, opzioni, checkpoint e mesh rigenerabili", "riproducibilita.json", "dipende dalle versioni delle librerie"),
        ("Validazione sperimentale", "FUORI da questo notebook: il confronto con le misure e l'incertezza dei dati (ASME V&V 20) richiedono i dati sperimentali", "non coperta", "va affrontata separatamente prima della sottomissione"),
    ]
    return pd.DataFrame(R, columns=['Critica del revisore', 'Risposta nell\'analisi', 'Evidenza (file)', 'Limite residuo'])


def riepilogo_robustezza(base):
    """Aggrega i test di robustezza della sensitivita': passo delle differenze finite, rumore assunto, mesh. Restituisce dict di DataFrame (dettaglio e sintesi)."""
    D, Rn, M = [], [], []
    for fam in FAMIGLIE:
        d0 = os.path.join(base, fam, 'casi')
        if not os.path.isdir(d0):
            continue
        for cid in sorted(os.listdir(d0)):
            d = os.path.join(d0, cid)
            f = os.path.join(d, 'S_delta.csv')
            if os.path.exists(f):
                x = pd.read_csv(f); x['famiglia'] = fam; x['caso'] = cid; D.append(x)
            f = os.path.join(d, 'S_rumore.csv')
            if os.path.exists(f):
                x = pd.read_csv(f); x['famiglia'] = fam; x['caso'] = cid; Rn.append(x)
            fs = os.path.join(d, 'S.csv'); alt = sorted(g for g in os.listdir(d) if g.startswith('S_L'))
            if os.path.exists(fs) and alt:
                a = pd.read_csv(fs); a = a[a.sonde == 'tutte'].set_index('parametro'); b = pd.read_csv(os.path.join(d, alt[0])); b = b[b.sonde == 'tutte'].set_index('parametro')
                for p_ in a.index:
                    M.append(dict(famiglia=fam, caso=cid, parametro=p_, effetto_mesh1=a.loc[p_, 'effetto_10pc_C'], effetto_mesh2=b.loc[p_, 'effetto_10pc_C'],
                                  sel_mesh1=bool(a.loc[p_, 'selezionato']), sel_mesh2=bool(b.loc[p_, 'selezionato'])))
    out = {}
    if D:
        Dd = pd.concat(D, ignore_index=True); out['delta'] = Dd
        out['delta_sintesi'] = Dd.groupby(['famiglia', 'parametro']).dif_rel_pc.agg(mediana_assoluta=lambda x: np.nanmedian(np.abs(x)), massimo_assoluto=lambda x: np.nanmax(np.abs(x))).reset_index()
    if Rn:
        Rr = pd.concat(Rn, ignore_index=True); out['rumore'] = Rr
        righe = []
        for (fam, s), g in Rr.groupby(['famiglia', 'rumore_K']):
            for p_ in dg.PARAMETRI_CAL:
                righe.append(dict(famiglia=fam, rumore_K=s, parametro=p_, selezionato_pc=100 * np.mean([p_ in str(x).split(',') for x in g.selezione])))
        out['rumore_sintesi'] = pd.DataFrame(righe)
    if M:
        Mm = pd.DataFrame(M); Mm['dif_rel_pc'] = 100 * (Mm.effetto_mesh1 - Mm.effetto_mesh2) / Mm.effetto_mesh2.replace(0, np.nan); Mm['stessa_selezione'] = Mm.sel_mesh1 == Mm.sel_mesh2; out['mesh'] = Mm
        out['mesh_sintesi'] = Mm.groupby(['famiglia', 'parametro']).agg(dif_mediana_pc=('dif_rel_pc', 'median'), dif_p90_assoluta_pc=('dif_rel_pc', lambda x: np.nanpercentile(np.abs(x), 90)), stessa_selezione_pc=('stessa_selezione', lambda x: 100 * x.mean())).reset_index()
    return out
