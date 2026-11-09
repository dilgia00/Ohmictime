"""
Analisi di verifica numerica e sensitivita' con MESH CONFORMI alle sfere (interfaccia netta).

Stesso protocollo del notebook con mesh strutturata, applicato a una famiglia di mesh tetraedriche generate con gmsh,
conformi alle sfere e raffinate all'interfaccia:
  A  convergenza spaziale del problema elettrico;
  B  convergenza spaziale del problema accoppiato (tempo di valutazione limitato prima dell'ebollizione);
  C  convergenza nel passo temporale (mesh fissa);
  S  sensitivita' locale scalata, sulle stesse mesh conformi (durata limitata al tempo di valutazione).
I file prodotti (A.csv, B_tempo_fisso.csv, C.csv, S.csv, riepilogo.json) hanno lo stesso formato del notebook strutturato:
aggregazione, tabelle di incertezza (GCI e minimi quadrati) e report si usano senza modifiche.

La dimensione rappresentativa delle mesh non strutturate e' h = (V / N_tetraedri)^(1/3); il rapporto di raffinamento fra mesh
successive non e' costante: il GCI ammette rapporti variabili (Celik et al., 2008) e il fit ai minimi quadrati non lo richiede.
Richiede diagnostica_ohmico.py, multifase_engine.py, gmsh e meshio.
"""
import logging
import os
import time

import numpy as np
import pandas as pd

logging.getLogger('skfem').setLevel(logging.ERROR)
import diagnostica_ohmico as dg                      # noqa: E402

OPZ_CONF = dict(
    lc_base_mm=8.0,           # dimensione globale degli elementi della mesh piu' grossolana [mm]
    fattore_lc=2 ** 0.5,      # rapporto fra le dimensioni di due livelli successivi
    rapporto_lci=0.5,         # dimensione all'interfaccia / dimensione globale
    livelli_conf=4,           # numero di livelli della famiglia
    max_nodi_conf=70000,      # i livelli oltre questo numero di nodi non vengono eseguiti (restano almeno 3 livelli)
    griglia_C=1,              # livello usato per lo studio del passo temporale
    max_nodi_S=8000,          # livello piu' fine con al massimo questi nodi per la sensitivita'
    cartella_mesh='meshes_conformi',   # cache delle mesh (esterna alla cartella dei risultati: i file .msh sono pesanti e rigenerabili)
    tentativi_mesh=3,
)


def opzioni(opz=None):
    o = dict(dg.OPZ_DEFAULT); o.update(OPZ_CONF); o.update(opz or {})
    o['riferimento_netto'] = 'no'
    return o


def livelli_lc(o):
    """Elenco [(lc, lci)] in metri, dalla mesh piu' grossolana alla piu' fine."""
    return [(o['lc_base_mm'] * 1e-3 / o['fattore_lc'] ** k, o['lc_base_mm'] * 1e-3 / o['fattore_lc'] ** k * o['rapporto_lci']) for k in range(o['livelli_conf'])]


def _chiave_cella(geo, config):
    return f"{int(round(geo['Lx'] * 1000))}x{int(round(geo['Ly'] * 1000))}x{int(round(geo['Lz'] * 1000))}_{config}"


def file_mesh(caso, k, o):
    sfx = f"N{int(o['nodi_obiettivo'][k])}" if o.get('nodi_obiettivo') else f"{o['lc_base_mm']:g}"
    return os.path.join(o['cartella_mesh'], f"{_chiave_cella(caso['geo'], caso['config'])}_L{k}_{sfx}.msh")


def mesh_livello(caso, k, o):
    """Mesh conforme del livello k per la geometria del caso (dipende da cella e configurazione, non dalla ricetta): generata una volta e riusata.
    Se 'nodi_obiettivo' e' presente la dimensione globale e' corretta fino a ottenere il numero di nodi del livello (entro 'tolleranza_nodi'); altrimenti si usa lc_base_mm."""
    f = file_mesh(caso, k, o); os.makedirs(os.path.dirname(f) or '.', exist_ok=True)
    if not os.path.exists(f):
        ultimo = None
        if o.get('nodi_obiettivo'):
            geo = caso['geo']; V = geo['Lx'] * geo['Ly'] * geo['Lz']; N = float(o['nodi_obiettivo'][k]); lc = o['kappa_lc'] * (V / N) ** (1.0 / 3.0)
            for t in range(o['tentativi_mesh'] + 2):
                try:
                    dg.mesh_conforme_gmsh(geo, lc, lc * o['rapporto_lci'], f); mesh, tags = dg._carica_conforme(f); n = mesh.p.shape[1]
                except Exception as e:
                    ultimo = e
                    if os.path.exists(f): os.remove(f)
                    lc *= 1.03; continue
                if abs(n / N - 1.0) <= o['tolleranza_nodi'] or t == o['tentativi_mesh'] + 1:
                    return mesh, tags
                os.remove(f); lc *= (n / N) ** (1.0 / 3.0)
            raise RuntimeError(f"generazione della mesh conforme (livello {k}) non riuscita: {ultimo}")
        lc, lci = livelli_lc(o)[k]
        for t in range(o['tentativi_mesh']):
            try:
                dg.mesh_conforme_gmsh(caso['geo'], lc * (1 + 0.02 * t), lci * (1 + 0.02 * t), f)
                break
            except Exception as e:           # gmsh puo' fallire su una dimensione particolare: si ritenta con una dimensione appena diversa
                ultimo = e
                if os.path.exists(f):
                    os.remove(f)
        else:
            raise RuntimeError(f"generazione della mesh conforme (livello {k}) non riuscita: {ultimo}")
    return dg._carica_conforme(f)


def contesto_conforme(caso, k, o):
    mesh, tags = mesh_livello(caso, k, o)
    ctx = dg.Contesto(caso['geo'], None, caso['p'], costruisci_filtro=False, mesh_ext=(mesh, tags))
    ctx.n_tet = int(mesh.t.shape[1]); ctx.livello = k
    return ctx


def analizza_caso_conforme(caso, cartella, opz):
    """Analisi completa di un caso su mesh conformi. Stessa uscita di diagnostica_ohmico.analizza_caso (file CSV e riepilogo.json)."""
    o = opzioni(opz)
    cid, geo, p = caso['id'], caso['geo'], caso['p']
    dirc = os.path.join(cartella, cid); os.makedirs(dirc, exist_ok=True)
    t0 = time.time()
    dt = o['dt']; t_fix = o['t_fisso']; T0c = p['T_iniziale']
    riep = dict(id=cid, ricetta=caso['ricetta'], config=caso['config'], cella=caso['cella'], sigma_m20=p['_sig_m20'], alfa_m=p['_alfa_m'], mesh_tipo='conforme')
    L = o['livelli_conf']
    # ---- tempo di valutazione: al piu' il tempo in cui T_max raggiunge T_limite_valutazione (il modello non include l'ebollizione) ----
    t_eval, limitato = t_fix, False
    if o.get('T_limite_valutazione'):
        ctxp = contesto_conforme(caso, min(1, L - 1), o)
        pp = dict(p); pp['dt'] = 5.0
        _, infp = dg.simula_diag(ctxp, pp, 1e9, T_crit=float(o['T_limite_valutazione']), durata_max=t_fix + 10, t_min_run=0.0)
        if infp['t_boil'] is not None and infp['t_boil'] < t_fix:
            t_eval, limitato = float(max(30.0, np.floor(infp['t_boil'] / 5.0) * 5.0)), True
        del ctxp
    riep.update(t_valutazione=t_eval, t_limitato_ebollizione=limitato, t_fisso=t_fix)
    # ---- A e B: famiglia di mesh conformi ----
    rA, rB, nodi = [], [], []
    for k in range(L):
        ctx = contesto_conforme(caso, k, o)
        N = int(ctx.basis.N)
        if N > o['max_nodi_conf'] and len(rA) >= 3:
            del ctx
            break
        e = dg.solo_elettrico(ctx, p); e.update(h_rep_mm=ctx.h_rep * 1000, n_nodi=N, n_tet=ctx.n_tet, livello=k, lc_mm=float('nan')); rA.append(e)
        serie, info = dg.simula_diag(ctx, p, 1e9, durata_max=t_eval + 2 * dt, t_min_run=t_eval)
        tf = dg.valori_a_tempo(serie, t_eval); tf.update(h_rep_mm=ctx.h_rep * 1000, n_nodi=N, n_tet=ctx.n_tet, livello=k)
        rB.append(tf); nodi.append(N)
        riep['chiusura_max'] = max(riep.get('chiusura_max', 0.0), float(serie.chiusura_energia.abs().max()))
        del ctx
    dfA, dfB = pd.DataFrame(rA), pd.DataFrame(rB)
    dfA.to_csv(os.path.join(dirc, 'A.csv'), index=False); dfB.to_csv(os.path.join(dirc, 'B_tempo_fisso.csv'), index=False)
    riep.update(A_ident_max=float((dfA.P_meno_VI.abs() / dfA.P).max()), A_sbil_max=float(dfA.sbilancio_I.max()), A_griglie=len(dfA), B_griglie=len(dfB), A_nodi_max=int(dfA.n_nodi.max()))
    # ---- C: passo temporale su un livello fisso (si dimezza ancora il passo se nessuno soddisfa la soglia) ----
    kC = min(o['griglia_C'], len(dfA) - 1)
    ctxC = contesto_conforme(caso, kC, o)
    dts = [dt * 4, dt * 2, dt, dt / 2]
    righeC = []

    def _un_dt(d_):
        pp = dict(p); pp['dt'] = float(d_)
        serie, info = dg.simula_diag(ctxC, pp, 1e9, durata_max=t_eval + 2 * float(d_), t_min_run=t_eval)
        tf = dg.valori_a_tempo(serie, t_eval); tf.update(dt=d_, stato=info['stato']); righeC.append(tf)
    for d_ in dts:
        _un_dt(d_)
    tol = {g: o['tol_abs_C'] for g in dg._QOI_T}
    extra = 0
    while True:
        dfC = pd.DataFrame(righeC)
        gC = [g for g in ['T_media_polp', 'T_media_pur', 'scarto_pol_pur', 'T_P1', 'T_P2', 'T_P3', 'E_in'] if g in dfC.columns and np.all(np.isfinite(dfC[g].values))]
        riseC = float(dfC['T_max'].values[np.argmin(dfC['dt'].values)] - T0c)
        vC = dg.valuta_convergenza(dfC, dfC['dt'].values / 1000.0, gC, ['T_max'], o, {g: riseC for g in dg._QOI_T}, tol)
        if vC['h_ess'] is not None or extra >= o.get('dt_livelli_extra', 3):
            break
        _un_dt(float(dfC['dt'].min()) / 2.0); extra += 1
    dfC.to_csv(os.path.join(dirc, 'C.csv'), index=False)
    riep.update(C_dt_livelli_extra=extra, C_nodi=int(ctxC.basis.N))
    del ctxC
    # ---- S: sensitivita' sulla mesh conforme piu' fine con al massimo max_nodi_S nodi; durata limitata al tempo di valutazione ----
    if o['esegui_S']:
        kS = 0
        for k, N in enumerate(dfA.n_nodi.values):
            if N <= o['max_nodi_S']:
                kS = k
        ctxS = contesto_conforme(caso, kS, o)
        nomi = sorted(ctxS.nomi_sonde)
        fase_p = dict(zip(ctxS.nomi_sonde, ctxS.Pr @ ctxS.phi_n))
        dur_S = float(min(o['durata_misure'], np.floor(t_eval / o['passo_misura']) * o['passo_misura']))

        def sonde_fn(theta, pb, tempi):
            pp = dg.applica_param(pb, theta); pp['dt'] = float(o['dt_S'])
            serie, _ = dg.simula_diag(ctxS, pp, 1e9, durata_max=dur_S + 2 * o['dt_S'], t_min_run=dur_S)
            M = np.vstack([np.interp(tempi, serie['t'].values, serie['T_' + n].values) for n in nomi])
            return M, {n: float(fase_p[n]) for n in nomi}
        tabS, X, selez = dg.sensibilita_locale(geo, p, None, o['dt_S'], dur_S, o['delta'], o['passo_misura'], o['sigma_misura'], o['se_max'], sonde_fn=sonde_fn)
        tabS.to_csv(os.path.join(dirc, 'S.csv'), index=False)
        riep.update(S_mesh=f"conforme L{kS} ({int(ctxS.basis.N)} nodi)", S_durata=dur_S, S_sel_tutte=','.join(selez.get('tutte', [])),
                    S_sel_purea=','.join(selez.get('purea', [])), S_sel_polpetta=','.join(selez.get('polpetta', [])))
        del ctxS
    riep['tempo_s'] = time.time() - t0
    dg._json_salva(riep, os.path.join(dirc, 'riepilogo.json'))
    # ---- indicatori di riepilogo (stessi del notebook strutturato) dai file appena scritti ----
    return dg.ricalcola_caso(dirc, cid, o, T0c)


def esegui_tutti_conforme(ricette_sel, config_sel, cartella='risultati_sensitivita_conforme', opz=None, campo_V_cm=4.0, T0=20.0, T_amb=21.0,
                          riprendi=True, verbose=True, cartella_drive=None):
    """Ciclo su tutte le combinazioni (ricetta, configurazione) con mesh conformi. Checkpoint e Drive come nel notebook strutturato."""
    o = opzioni(opz)
    return dg.esegui_tutti(ricette_sel, config_sel, cartella, o, campo_V_cm, T0, T_amb, riprendi, verbose, cartella_drive, analizza=analizza_caso_conforme)
