"""
Analisi di sensitivita' (numerica e parametrica) del modello elettro-termico del riscaldamento ohmico
multifase (purea + polpette), basata su grandezze di trasporto.

Richiede multifase_engine.py (stessa cartella). Non tocca il codice dell'applicazione Streamlit.

Contenuto
  1. Costruzione di mesh strutturate a numero di intervalli prescritto (serie di raffinamento a rapporto costante).
  2. Campo di fase (rampa proporzionale a h oppure fissa; correzione opzionale del volume delle sfere).
  3. Solutore accoppiato strumentato (copia fedele di multifase_engine.simula) che a ogni passo calcola:
       potenza P, correnti ai due elettrodi da residuo, potenza per fase, q medio in regioni geometriche,
       percentili di q, bilancio energetico discreto, energia scambiata purea-polpette, perdite superficiali,
       temperatura del punto freddo (media su una sfera di raggio fisso), temperature alle sonde.
  4. Istante di processo tau*: il punto freddo della polpetta raggiunge la temperatura di cottura.
  5. Analisi di convergenza: procedura GCI (Celik et al., 2008) e fit ai minimi quadrati (Eca e Hoekstra).
  6. Studi: spaziale elettrico, spaziale accoppiato, temporale, interfaccia; piano fattoriale 2^k; riferimento conforme (gmsh).
"""
import itertools
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
from skfem import Basis, BilinearForm, ElementTetP1, FacetBasis, LinearForm, MeshTet
from skfem.helpers import dot, grad

import logging
logging.getLogger('skfem').setLevel(logging.ERROR)      # avvisi informativi sulla conversione delle mesh (nessun effetto sui risultati)

import multifase_engine as me

# ---------------------------------------------------------------------------------------------------------
# Costanti di riferimento (fonti nella guida)
# ---------------------------------------------------------------------------------------------------------
T_COTTURA_C = 74.0      # pollame macinato, USDA/FSIS: 165 F = 74 C (foodsafety.gov, "Cook to a Safe Minimum Internal Temperature")
T_BOLLITURA_C = 100.0   # oltre questa temperatura il modello non e' valido (nessuna ebollizione)


# ---------------------------------------------------------------------------------------------------------
# 1. Mesh e campo di fase
# ---------------------------------------------------------------------------------------------------------
def costruisci_mesh_n(geo, n):
    """Mesh tensoriale con n = (ex, ey, ez) INTERVALLI per direzione (nodi = n + 1). Ogni cella -> 6 tetraedri."""
    ex, ey, ez = n
    mesh = MeshTet.init_tensor(np.linspace(0, geo['Lx'], ex + 1), np.linspace(0, geo['Ly'], ey + 1),
                               np.linspace(0, geo['Lz'], ez + 1))
    basis = Basis(mesh, ElementTetP1())
    h_eff = max(geo['Lx'] / ex, geo['Ly'] / ey, geo['Lz'] / ez)                  # passo massimo (usato dal motore)
    # h_rep = (V / N_celle_esaedriche)^(1/3): dimensione rappresentativa (Celik et al., 2008, eq. 1, con N = celle esaedriche)
    h_rep = (geo['Lx'] * geo['Ly'] * geo['Lz'] / ((ex + 1) * (ey + 1) * (ez + 1))) ** (1.0 / 3.0)      # h = (V / N_nodi)^(1/3): stessa definizione per le due famiglie
    return mesh, basis, h_eff, h_rep


def campo_fase(geo, basis, h_eff, rampa='prop', w_fissa=None, corr_volume=False):
    """Frazione di fase ai punti di quadratura e ai nodi.
    rampa='prop': larghezza w = h_eff/2 (come nel motore); rampa='fissa': w = w_fissa [m] indipendente dalla mesh.
    corr_volume=True: il raggio effettivo R_eff e' scelto in modo che l'integrale di phi eguagli il volume esatto delle sfere."""
    w = 0.5 * h_eff if rampa == 'prop' else float(w_fissa)
    Xq = basis.global_coordinates().value
    dx = basis.dx
    R = geo['R']
    n_sfere = len(geo['centri'])
    R_eff = R
    if corr_volume and n_sfere:
        V_ex = n_sfere * 4.0 / 3.0 * np.pi * R ** 3
        lo, hi = R - 2 * w - 0.5 * h_eff, R + 2 * w + 0.5 * h_eff
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            V = float((me.frazione_fase(Xq, geo['centri'], mid, w) * dx).sum())
            lo, hi = (mid, hi) if V < V_ex else (lo, mid)
        R_eff = 0.5 * (lo + hi)
    phi_q = me.frazione_fase(Xq, geo['centri'], R_eff, w)
    phi_n = me.frazione_fase(basis.doflocs, geo['centri'], R_eff, w)
    return phi_q, phi_n, dict(R_eff=R_eff, w=w, volume_phi=float((phi_q * dx).sum()),
                              volume_esatto=n_sfere * 4.0 / 3.0 * np.pi * R ** 3)


# ---------------------------------------------------------------------------------------------------------
# 2. Strumenti per le grandezze di interesse
# ---------------------------------------------------------------------------------------------------------
def percentile_pesato(valori, pesi, perc):
    """Percentili (0-100) pesati. valori, pesi: array 1D."""
    idx = np.argsort(valori)
    v, w = valori[idx], pesi[idx]
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return np.interp(np.atleast_1d(perc) / 100.0, c, v)


def regioni_geometriche(geo, Xq, phi_q, frac_tubo=0.5):
    """Pesi (ai punti di quadratura) di regioni definite dalla GEOMETRIA (non dai nodi della mesh); solo purea.
       'strati_z': purea con |z - z0| > R  (fasce sopra e sotto le sfere);
       'asse':     purea entro un tubo di raggio frac_tubo*R attorno all'asse della corrente (tratti tra sfere ed elettrodi)."""
    R = geo['R']
    y0, z0 = geo['Ly'] / 2.0, geo['Lz'] / 2.0
    pur = 1.0 - phi_q
    strati = (np.abs(Xq[2] - z0) > R).astype(float) * pur
    asse = (((Xq[1] - y0) ** 2 + (Xq[2] - z0) ** 2) < (frac_tubo * R) ** 2).astype(float) * pur
    return dict(strati_z=strati, asse=asse)


def operatore_filtro_freddo(geo, Xq, dx, phi_q, r_f, fase='polpetta'):
    """Operatore lineare (matrice sparsa) che da T ai punti di quadratura restituisce la media pesata di T su sfere di raggio
    fisso r_f centrate su un reticolo di passo r_f/2 dentro la fase scelta. Il peso e' dx*phi (polpetta) o dx*(1-phi) (purea).
    Restituisce (A, centri)."""
    pts = Xq.reshape(3, -1).T
    w_fase = (phi_q if fase == 'polpetta' else 1.0 - phi_q).ravel() * dx.ravel()
    passo = r_f / 2.0
    cand = []
    if fase == 'polpetta' and geo['centri']:
        for c in geo['centri']:
            g = np.arange(-geo['R'], geo['R'] + 1e-12, passo)
            G = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
            G = G[(G ** 2).sum(1) <= geo['R'] ** 2] + np.array(c)
            cand.append(G)
    else:
        gx = np.arange(passo, geo['Lx'], passo); gy = np.arange(passo, geo['Ly'], passo); gz = np.arange(passo, geo['Lz'], passo)
        cand.append(np.array(np.meshgrid(gx, gy, gz, indexing='ij')).reshape(3, -1).T)
    centri = np.vstack(cand)
    tree = cKDTree(pts)
    nb = tree.query_ball_point(centri, r_f)
    righe, colonne, valori = [], [], []
    for i, lst in enumerate(nb):
        if not lst:
            continue
        lst = np.asarray(lst)
        w = w_fase[lst]
        s = w.sum()
        if s <= 0:
            continue
        righe.append(np.full(len(lst), i)); colonne.append(lst); valori.append(w / s)
    A = coo_matrix((np.concatenate(valori), (np.concatenate(righe), np.concatenate(colonne))),
                   shape=(len(centri), pts.shape[0])).tocsr()
    ok = np.asarray(A.sum(axis=1)).ravel() > 0
    return A[ok], centri[ok]


def _entalpia_tabella(p, T_grid):
    """Entalpia volumetrica specifica H^(T) = int_{T0}^{T} rho*cp dT' per fase (J/m3), da tabelle di Choi-Okos."""
    tab = me._tabelle_proprieta(p, T_grid)
    H = {}
    for f in ('m', 'p'):
        rc = tab[f][1]
        H[f] = np.concatenate([[0.0], np.cumsum(0.5 * (rc[1:] + rc[:-1]) * np.diff(T_grid))])
    return H


# ---------------------------------------------------------------------------------------------------------
# 3. Quantita' elettriche comuni (usate sia dal solutore accoppiato sia dal problema solo elettrico)
# ---------------------------------------------------------------------------------------------------------
class Contesto:
    """Raccoglie forme, basi, regioni e operatori costruiti una sola volta per una mesh/geometria/parametri."""

    def __init__(self, geo, n, p, rampa='prop', w_fissa=None, corr_volume=False, frac_tubo=0.5, r_filtro=3e-3, costruisci_filtro=True, mesh_ext=None):
        self.geo, self.n, self.p = geo, n, p
        if mesh_ext is None:
            self.mesh, self.basis, self.h_eff, self.h_rep = costruisci_mesh_n(geo, n)
            b = self.basis
            self.phi_q, self.phi_n, self.info_fase = campo_fase(geo, b, self.h_eff, rampa, w_fissa, corr_volume)
        else:                                              # mesh conforme (interfaccia netta): phi vale 0 o 1 per elemento
            mesh, tags = mesh_ext
            self.mesh, self.basis, self.h_eff = mesh, Basis(mesh, ElementTetP1()), np.nan
            b = self.basis
            self.h_rep = (geo['Lx'] * geo['Ly'] * geo['Lz'] / mesh.p.shape[1]) ** (1.0 / 3.0)      # h = (V / N_nodi)^(1/3): stessa definizione per le due famiglie
            phi_el = (np.asarray(tags) == 2).astype(float)
            self.phi_q = phi_el[:, None] * np.ones_like(b.dx)
            vol = b.dx.sum(axis=1); num = np.zeros(b.N); den = np.zeros(b.N)
            for k in range(4):
                np.add.at(num, mesh.t[k], vol * phi_el / 4.0); np.add.at(den, mesh.t[k], vol / 4.0)
            self.phi_n = num / den
            V_ex = len(geo['centri']) * 4.0 / 3.0 * np.pi * geo['R'] ** 3
            self.info_fase = dict(R_eff=geo['R'], w=0.0, volume_phi=float((self.phi_q * b.dx).sum()), volume_esatto=V_ex)
        self.Xq = b.global_coordinates().value
        self.dx = b.dx
        self.V_rms = p['V_rms'] if 'V_rms' in p else p['E_rms_V_cm'] * 100.0 * geo['Lx']

        @BilinearForm
        def laplace(u, v, w):
            return w['sigma'] * dot(grad(u), grad(v))

        @LinearForm
        def sorgente(v, w):
            return w['q'] * v

        @BilinearForm
        def massa(u, v, w):
            return w['rhocp'] * u * v

        @BilinearForm
        def conduzione(u, v, w):
            return w['k'] * dot(grad(u), grad(v))

        @BilinearForm
        def massa_unitaria(u, v, w):
            return u * v

        @BilinearForm
        def robin_a(u, v, w):
            return w['h'] * u * v

        @LinearForm
        def robin_l(v, w):
            return w['g'] * v
        self.laplace, self.sorgente, self.massa, self.conduzione, self.robin_a, self.robin_l = laplace, sorgente, massa, conduzione, robin_a, robin_l
        self.ml = np.asarray(massa_unitaria.assemble(b).sum(axis=1)).ravel()
        self.facce = self.mesh.facets_satisfying(lambda x: np.isclose(x[2], geo['Lz']))
        self.fb = FacetBasis(self.mesh, b.elem, facets=self.facce)
        self.L_car = geo['Lx'] * geo['Ly'] / (2.0 * (geo['Lx'] + geo['Ly']))
        self.dofs_sx = b.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
        self.dofs_dx = b.get_dofs(lambda x: np.isclose(x[0], geo['Lx'])).all()
        self.D = np.concatenate([self.dofs_sx, self.dofs_dx])
        self.V_init = 1.0 - b.doflocs[0] / geo['Lx']
        self.b_zero = b.zeros()
        nomi = list(geo['sonde'].keys())
        self.nomi_sonde = nomi
        self.Pr = b.probes(np.array([geo['sonde'][k] for k in nomi]).T).tocsr()
        self.reg = regioni_geometriche(geo, self.Xq, self.phi_q, frac_tubo)
        self.fase_cold = 'polpetta' if geo['centri'] else 'purea'
        self.Af, self.centri_filtro = (operatore_filtro_freddo(geo, self.Xq, self.dx, self.phi_q, r_filtro, self.fase_cold)
                                       if costruisci_filtro else (None, None))
        self.T_grid = np.arange(-10.0, 160.1, 1.0)
        self.tab = me._tabelle_proprieta(p, self.T_grid)
        self.Hs = _entalpia_tabella(p, self.T_grid)

    # ---- proprieta' termiche ai punti di quadratura ----
    def campi_termici(self, T_q):
        t, tab, phi = self.T_grid, self.tab, self.phi_q
        k = (1 - phi) * np.interp(T_q, t, tab['m'][0]) + phi * np.interp(T_q, t, tab['p'][0])
        rc = (1 - phi) * np.interp(T_q, t, tab['m'][1]) + phi * np.interp(T_q, t, tab['p'][1])
        return k, rc

    # ---- problema elettrico a potenziale unitario + grandezze derivate ----
    def elettrico(self, sig_q, percentili=(1, 99)):
        b, dx, phi = self.basis, self.dx, self.phi_q
        A = self.laplace.assemble(b, sigma=sig_q)
        Vu = me._risolvi_elettrico(A, self.b_zero, self.V_init, self.D)
        gV = b.interpolate(Vu).grad
        q_q = self.V_rms ** 2 * sig_q * (gV[0] ** 2 + gV[1] ** 2 + gV[2] ** 2)
        res = A @ Vu                                       # residuo del sistema NON vincolato (flussi nodali)
        I_L, I_R = float(res[self.dofs_sx].sum()), float(res[self.dofs_dx].sum())   # [S] per potenziale unitario
        P = float((q_q * dx).sum())
        P_p = float((q_q * dx * phi).sum())
        out = dict(P=P, G_ris=I_L, I_L=self.V_rms * I_L, I_R=self.V_rms * I_R, P_meno_VI=P - self.V_rms ** 2 * I_L,
                   sbilancio_I=abs(I_L + I_R) / max(abs(I_L), 1e-300), P_p=P_p, P_m=P - P_p, frac_P_p=P_p / P if P > 0 else np.nan)
        for nome, w in self.reg.items():
            ww = w * dx
            out['q_' + nome] = float((q_q * ww).sum() / ww.sum()) if ww.sum() > 0 else np.nan
        pur = ((1 - phi) * dx).ravel()
        pq = percentile_pesato(q_q.ravel(), dx.ravel(), percentili)
        for pc, v in zip(percentili, pq):
            out[f'q_p{int(pc):02d}'] = float(v)
        if (pur > 0).any():
            pq = percentile_pesato(q_q.ravel()[pur > 0], pur[pur > 0], [99])
            out['q_p99_purea'] = float(pq[0])
        out['q_media_polp'] = float((q_q * dx * phi).sum() / (dx * phi).sum()) if (dx * phi).sum() > 0 else np.nan
        out['q_media_pur'] = float((q_q * dx * (1 - phi)).sum() / (dx * (1 - phi)).sum())
        return out, q_q, Vu


# ---------------------------------------------------------------------------------------------------------
# 4. Solutore accoppiato strumentato
# ---------------------------------------------------------------------------------------------------------
def simula_diag(ctx, p, T_target=T_COTTURA_C, T_crit=T_BOLLITURA_C, durata_max=None, ferma_a_tau=True, verbose=False, t_min_run=0.0, snap_n=0, avanzamento=None):
    """Copia fedele di multifase_engine.simula con diagnostica. Restituisce (DataFrame serie, dizionario info).
    Le grandezze elettriche e di stato sono riferite a t_n (stato T^n); le energie cumulate a t_n.
    Bilancio discreto esatto (coefficienti congelati a t_n):
        int rho*cp^n (T^{n+1}-T^n) dV  =  dt * ( P^n - Q_perdite^{n+1} ),   Q_perdite^{n+1} = int [q0 + h_lin (T^{n+1}-T^n)] dGamma."""
    geo, b, mesh, dx, phi = ctx.geo, ctx.basis, ctx.mesh, ctx.dx, ctx.phi_q
    ctx.V_rms = p['V_rms'] if 'V_rms' in p else p['E_rms_V_cm'] * 100.0 * geo['Lx']     # il contesto puo' essere riusato con tensioni diverse
    dt = p['dt']
    tmax = durata_max if durata_max is not None else p['durata']
    n_passi = max(1, int(round(tmax / dt)))
    prop_dip_T = p['prop_modo'] == 'choi_okos_T'
    ctx.tab = me._tabelle_proprieta(p, ctx.T_grid)
    ctx.Hs = _entalpia_tabella(p, ctx.T_grid)

    phi_n = ctx.phi_n
    if p.get('T_iniziale_sfere') is not None:          # temperatura iniziale diversa per ogni polpetta (ordine dei centri in geo['centri']): ogni nodo prende quella della polpetta piu' vicina
        Ts = np.asarray(p['T_iniziale_sfere'], float); Cc = np.asarray(geo['centri'], float); Pn = ctx.mesh.p
        idx = np.argmin(((Pn[:, :, None] - Cc.T[:, None, :]) ** 2).sum(0), axis=1)
        T = (1.0 - phi_n) * p['T_iniziale'] + phi_n * Ts[idx]
    else:
        T = (1.0 - phi_n) * p['T_iniziale'] + phi_n * p.get('T_iniziale_polpetta', p['T_iniziale'])
    T_q0 = b.interpolate(T).value
    k_q, rhocp_q = ctx.campi_termici(T_q0)
    M = ctx.massa.assemble(b, rhocp=rhocp_q)
    A0 = (M / dt + ctx.conduzione.assemble(b, k=k_q)).tocsr()

    righe = []
    E_in = E_st = E_loss = E_st_p = E_st_m = E_pm = 0.0
    snaps = []
    snap_idx = set(int(round(x)) for x in np.linspace(0, n_passi, snap_n)) if snap_n and snap_n > 1 else set()
    stato, t_stato = 'completo', None
    t_tau = t_boil = None
    for n in range(n_passi + 1):
        t = n * dt
        T_q = b.interpolate(T).value
        sig_q = me.sigma_campo(T_q, phi, p)
        el, q_q, Vu = ctx.elettrico(sig_q)
        # --- diagnostica di stato a t_n ---
        r = dict(t=t, **el)
        r.update(T_max=float(T.max()), T_min=float(T.min()),
                 T_polp_min_nodo=float(T[phi_n > 0.5].min()) if (phi_n > 0.5).any() else np.nan,
                 T_media_polp=float((T @ (ctx.ml * phi_n)) / (ctx.ml * phi_n).sum()) if phi_n.sum() > 0 else np.nan,
                 T_media_pur=float((T @ (ctx.ml * (1 - phi_n))) / (ctx.ml * (1 - phi_n)).sum()))
        if ctx.Af is not None:
            Tf = ctx.Af @ T_q.ravel()
            j = int(np.argmin(Tf))
            r.update(T_freddo=float(Tf[j]), x_freddo=float(ctx.centri_filtro[j, 0]), y_freddo=float(ctx.centri_filtro[j, 1]),
                     z_freddo=float(ctx.centri_filtro[j, 2]), T_caldo_filtrata=float(Tf.max()))
        Tp = ctx.Pr @ T
        for i, nm in enumerate(ctx.nomi_sonde):
            r['T_' + nm] = float(Tp[i])
        r.update(E_in=E_in, E_stored=E_st, E_loss=E_loss, E_stored_p=E_st_p, E_stored_m=E_st_m, E_pm=E_pm)
        r['chiusura_energia'] = (E_in - E_st - E_loss) / E_in if E_in > 0 else 0.0
        righe.append(r)
        if n in snap_idx: snaps.append((t, T.copy()))
        if avanzamento is not None and n % max(1, n_passi // 20) == 0: avanzamento(n / max(1, n_passi))
        if verbose:
            print(f"t={t:6.0f}  P={el['P']:.1f}  Tmax={r['T_max']:.1f}  Tfreddo={r.get('T_freddo', np.nan):.1f}")
        # --- eventi e criteri di arresto ---
        if t_tau is None and r.get('T_freddo', -1e9) >= T_target:
            t_tau = t
        if t_boil is None and r['T_max'] >= T_crit:
            t_boil = t
        if ferma_a_tau and t >= t_min_run and (t_tau is not None or t_boil is not None):
            break
        if n == n_passi:
            break
        # --- passo termico (identico al motore) ---
        if prop_dip_T and n > 0:
            k_q, rhocp_q = ctx.campi_termici(T_q)
            M = ctx.massa.assemble(b, rhocp=rhocp_q)
            A0 = (M / dt + ctx.conduzione.assemble(b, k=k_q)).tocsr()
        b_q = ctx.sorgente.assemble(b, q=q_q)
        Tf_s = ctx.fb.interpolate(T).value
        q0_f, h_lin = me.flusso_e_derivata(Tf_s, p, ctx.L_car)
        K_r = ctx.robin_a.assemble(ctx.fb, h=h_lin)
        b_r = ctx.robin_l.assemble(ctx.fb, g=h_lin * Tf_s - q0_f)
        A = (A0 + K_r).tocsr()
        Dinv = diags(1.0 / A.diagonal())
        T_new, info = cg(A, M @ T / dt + b_q + b_r, x0=T, M=Dinv, rtol=me.RTOL_TERMICO, maxiter=500)
        if info != 0:
            raise RuntimeError("Il solutore termico non e' converso.")
        # --- energie (stesso schema, coefficienti congelati) ---
        dTq = b.interpolate(T_new).value - T_q
        dE = float((rhocp_q * dTq * dx).sum())
        dE_p = float((rhocp_q * dTq * dx * phi).sum())
        Tf_new = ctx.fb.interpolate(T_new).value
        Q_loss = float(((q0_f + h_lin * (Tf_new - Tf_s)) * ctx.fb.dx).sum())
        E_in += dt * el['P']
        E_st += dE
        E_st_p += dE_p
        E_st_m += dE - dE_p
        E_loss += dt * Q_loss
        E_pm += dt * el['P_p'] - dE_p            # energia ceduta dalle polpette alla purea (negativa se le polpette ricevono calore)
        T = T_new
    if t_tau is not None and (t_boil is None or t_tau <= t_boil):
        stato, t_stato = 'tau_raggiunto', t_tau
    elif t_boil is not None:
        stato, t_stato = 'bollitura_prima_di_tau', t_boil
    else:
        stato, t_stato = 'tau_non_raggiunto', None
    serie = pd.DataFrame(righe)
    info = dict(snap=snaps, stato=stato, t_stato=t_stato, t_tau=t_tau, t_boil=t_boil, n_nodi=int(b.N), n_el=int(mesh.t.shape[1]), h_eff=ctx.h_eff, h_rep=ctx.h_rep, dt=dt,
                **{k: v for k, v in ctx.info_fase.items()})
    return serie, info


def valori_a_tau(serie, T_target=T_COTTURA_C, colonna_T='T_freddo'):
    """Interpola linearmente nel tempo tutte le colonne numeriche all'istante tau* in cui colonna_T raggiunge T_target.
    Restituisce (dict, stato). Se non raggiunto: dict vuoto."""
    T = serie[colonna_T].values
    idx = np.where(T >= T_target)[0]
    if len(idx) == 0:
        return {}, 'tau_non_raggiunto'
    i = idx[0]
    if i == 0:
        f, i0 = 0.0, 0
    else:
        i0 = i - 1
        f = (T_target - T[i0]) / (T[i] - T[i0])
    out = {}
    for c in serie.columns:
        v0, v1 = serie[c].iloc[i0], serie[c].iloc[i]
        out[c] = float(v0 + f * (v1 - v0))
    out['tau_star'] = out.pop('t')
    out['margine_ebollizione'] = T_BOLLITURA_C - out['T_max']
    out['scarto_pol_pur'] = out['T_media_pur'] - out['T_media_polp']
    out['T_max_meno_T_min'] = out['T_max'] - out['T_min']
    out['frac_E_perdite'] = out['E_loss'] / out['E_in'] if out['E_in'] > 0 else np.nan
    return out, 'ok'


def solo_elettrico(ctx, p, T_uniforme=None):
    """Problema elettrico a conducibilita' prescritta (T uniforme per fase): costo minimo, isola l'errore spaziale elettrico."""
    phi_n = ctx.phi_n
    T0m = p['T_iniziale'] if T_uniforme is None else T_uniforme
    T0p = p.get('T_iniziale_polpetta', T0m) if T_uniforme is None else T_uniforme
    T = (1.0 - phi_n) * T0m + phi_n * T0p
    sig_q = me.sigma_campo(ctx.basis.interpolate(T).value, ctx.phi_q, p)
    el, q_q, Vu = ctx.elettrico(sig_q)
    el.update(volume_phi=ctx.info_fase['volume_phi'], volume_esatto=ctx.info_fase['volume_esatto'],
              errore_volume=ctx.info_fase['volume_phi'] / ctx.info_fase['volume_esatto'] - 1.0 if ctx.info_fase['volume_esatto'] else 0.0)
    return el


# ---------------------------------------------------------------------------------------------------------
# 5. Analisi di convergenza
# ---------------------------------------------------------------------------------------------------------
def gci_celik(h, f, p_min=0.5, p_max=2.0, Fs=1.25):
    """Procedura di Celik et al. (2008) su tre griglie. h, f: sequenze (fine, media, grossolana).
    p osservato da iterazione a punto fisso (rapporto di raffinamento non costante ammesso); poi p limitato in [p_min, p_max]
    SOLO per il calcolo di GCI (protezione; l'ordine osservato e' comunque riportato)."""
    h1, h2, h3 = h
    f1, f2, f3 = f
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out = dict(r21=r21, r32=r32, e21=e21, e32=e32)
    if e21 == 0 or e32 == 0 or not np.isfinite(e21 * e32):
        out.update(R=np.nan, regime='indefinito', p_oss=np.nan, p_uso=np.nan, f_ext=np.nan, GCI_fine=np.nan, e_a=np.nan)
        return out
    R = e21 / e32
    s = np.sign(e32 / e21)
    p = 1.0
    for _ in range(500):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        pn = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if not np.isfinite(pn) or abs(pn - p) < 1e-10:
            p = pn
            break
        p = pn
    regime = 'monotona' if 0 < R < 1 else ('oscillante' if R < 0 else 'divergente')
    p_uso = float(np.clip(p, p_min, p_max)) if np.isfinite(p) else np.nan
    f_ext = (r21 ** p_uso * f1 - f2) / (r21 ** p_uso - 1)
    e_a = abs((f1 - f2) / f1) if f1 != 0 else np.nan
    gci = Fs * e_a / (r21 ** p_uso - 1) if np.isfinite(e_a) else np.nan
    out.update(R=R, regime=regime, p_oss=p, p_uso=p_uso, f_ext=f_ext, e_a=e_a, GCI_fine=gci)
    return out


def fit_potenza(h, f, p_fissi=None):
    """Fit f_i = f0 + a*h_i^p ai minimi quadrati (p libero se p_fissi e' None). Restituisce lista di dict (uno per p)."""
    h, f = np.asarray(h, float), np.asarray(f, float)
    griglia = np.linspace(0.2, 4.0, 761) if p_fissi is None else np.atleast_1d(p_fissi)
    best = None
    for p in griglia:
        X = np.column_stack([np.ones_like(h), h ** p])
        coef, *_ = np.linalg.lstsq(X, f, rcond=None)
        res = f - X @ coef
        rss = float((res ** 2).sum())
        if best is None or rss < best['rss']:
            best = dict(p=float(p), f0=float(coef[0]), a=float(coef[1]), rss=rss)
    n = len(h)
    best['Us'] = float(np.sqrt(best['rss'] / max(n - (3 if p_fissi is None else 2), 1)))
    best['delta_RE_fine'] = best['a'] * h.min() ** best['p']
    return best


def incertezza_ls(h, f, Fs=1.25, Fs_fallback=3.0):
    """Incertezza numerica sulla griglia piu' fine con fit ai minimi quadrati (>=4 griglie), nello spirito di Eca e Hoekstra (2014).
    Per 0.95 <= p < 2.05 si usa U = 1.25 (|delta_RE| + Us) (regola riportata nella presentazione dei due autori al Workshop CFD di Lisbona, 2008).
    Fuori da questo intervallo si ripiega, in modo conservativo, su un fit a p fissato (1 se p<0.95, 2 se p>=2.05) con U = 3 (|delta_RE| + Us)
    (scelta del presente codice, da verificare sul lavoro originale)."""
    if len(h) < 4:
        return dict(p_ls=np.nan, f0_ls=np.nan, U_ls=np.nan, U_rel=np.nan, ramo='griglie<4')
    lib = fit_potenza(h, f)
    if 0.95 <= lib['p'] < 2.05:
        U = Fs * (abs(lib['delta_RE_fine']) + lib['Us']); ramo = 'libero'; fit = lib
    else:
        pfix = 1.0 if lib['p'] < 0.95 else 2.0
        fit = fit_potenza(h, f, [pfix]); U = Fs_fallback * (abs(fit['delta_RE_fine']) + fit['Us']); ramo = f'fallback p={pfix:g}'
    f_fine = float(np.asarray(f)[np.argmin(h)])
    return dict(p_ls=lib['p'], f0_ls=fit['f0'], U_ls=U, U_rel=abs(U / f_fine) if f_fine != 0 else np.nan, ramo=ramo)


def tabella_convergenza(h, valori, grandezze=None, verbose=True):
    """h: lista di dimensioni rappresentative (qualsiasi ordine); valori: DataFrame (una riga per griglia) con le grandezze in colonna.
    Restituisce una tabella con p osservato, regime, valore estrapolato, GCI e incertezza ai minimi quadrati per ogni grandezza."""
    h = np.asarray(h, float)
    ordine = np.argsort(h)                       # dalla piu' fine alla piu' grossolana
    h = h[ordine]
    V = valori.iloc[ordine].reset_index(drop=True)
    righe = []
    for c in (grandezze or [c for c in V.columns if pd.api.types.is_numeric_dtype(V[c])]):
        f = V[c].values.astype(float)
        if len(f) < 3 or not np.all(np.isfinite(f)):
            continue
        g = gci_celik(h[:3], f[:3])
        ls = incertezza_ls(h, f)
        # controllo dell'intervallo asintotico con le prime quattro griglie (se disponibili)
        asint = np.nan
        if len(f) >= 4:
            g2 = gci_celik(h[1:4], f[1:4])
            if np.isfinite(g['GCI_fine']) and np.isfinite(g2['GCI_fine']) and g['GCI_fine'] > 0:
                asint = g2['GCI_fine'] / (g['r21'] ** g['p_uso'] * g['GCI_fine'])
        righe.append(dict(grandezza=c, f_fine=f[0], f_ext=g['f_ext'], p_osservato=g['p_oss'], regime=g['regime'], R=g['R'],
                          GCI_fine_percento=100 * g['GCI_fine'] if np.isfinite(g['GCI_fine']) else np.nan,
                          rapporto_asintotico=asint, p_LS=ls['p_ls'], U_LS_rel_percento=100 * ls['U_rel'] if np.isfinite(ls['U_rel']) else np.nan,
                          ramo_LS=ls['ramo']))
    return pd.DataFrame(righe)


# ---------------------------------------------------------------------------------------------------------
# 6. Studi
# ---------------------------------------------------------------------------------------------------------
GRIGLIE_R2 = [(10, 5, 4), (20, 10, 8), (40, 20, 16), (80, 40, 32)]      # rapporto di raffinamento 2 (cella UNISA 100 x 51 x 37 mm)

QOI_ELETTRICHE = ['P', 'I_L', 'frac_P_p', 'q_strati_z', 'q_asse', 'q_media_polp', 'q_media_pur', 'q_p01', 'q_p99', 'q_p99_purea']
QOI_TAU = ['tau_star', 'T_max', 'scarto_pol_pur', 'T_max_meno_T_min', 'P', 'frac_P_p', 'q_strati_z', 'q_asse', 'q_p99',
           'E_in', 'E_pm', 'E_loss', 'frac_E_perdite', 'T_P1', 'T_P2', 'T_P3', 'margine_ebollizione']


def studio_elettrico(geo, griglie, p, rampa='prop', w_fissa=None, corr_volume=False, frac_tubo=0.5, verbose=True):
    """Serie di raffinamento del solo problema elettrico (sigma prescritta)."""
    righe, h_rep = [], []
    for n in griglie:
        t0 = time.time()
        ctx = Contesto(geo, n, p, rampa, w_fissa, corr_volume, frac_tubo, costruisci_filtro=False)
        r = solo_elettrico(ctx, p)
        r.update(n_nodi=int(ctx.basis.N), h_rep_mm=ctx.h_rep * 1000, h_eff_mm=ctx.h_eff * 1000, rampa=rampa, corr_volume=corr_volume,
                 tempo_s=time.time() - t0)
        righe.append(r); h_rep.append(ctx.h_rep)
        if verbose:
            print(f"  griglia {n}: {r['n_nodi']} nodi, h_rep={ctx.h_rep * 1000:.2f} mm, P={r['P']:.3f} W, P_p/P={r['frac_P_p']:.4f}, "
                  f"sbilancio I={r['sbilancio_I']:.1e}, ({r['tempo_s']:.1f} s)")
    df = pd.DataFrame(righe)
    return df, np.array(h_rep)


def studio_accoppiato(geo, griglie, p, T_target=T_COTTURA_C, rampa='prop', w_fissa=None, corr_volume=False, r_filtro=3e-3,
                      durata_max=1500.0, verbose=True):
    """Serie di raffinamento del problema accoppiato; grandezze a tau*. Restituisce (tabella a tau*, h_rep, serie per griglia, info)."""
    righe, h_rep, serie_l, info_l = [], [], [], []
    for n in griglie:
        t0 = time.time()
        ctx = Contesto(geo, n, p, rampa, w_fissa, corr_volume, r_filtro=r_filtro)
        serie, info = simula_diag(ctx, p, T_target=T_target, durata_max=durata_max)
        val, stato = valori_a_tau(serie, T_target)
        val.update(n_nodi=info['n_nodi'], h_rep_mm=ctx.h_rep * 1000, stato=info['stato'], tempo_calcolo_s=time.time() - t0)
        righe.append(val); h_rep.append(ctx.h_rep); serie_l.append(serie); info_l.append(info)
        if verbose:
            print(f"  griglia {n}: {info['n_nodi']} nodi, stato={info['stato']}, tau*={val.get('tau_star', np.nan):.1f} s, "
                  f"chiusura energia max={serie['chiusura_energia'].abs().max():.1e}, ({val['tempo_calcolo_s']:.0f} s)")
    return pd.DataFrame(righe), np.array(h_rep), serie_l, info_l


def studio_temporale(geo, n, p, dts, T_target=T_COTTURA_C, r_filtro=3e-3, durata_max=1500.0, verbose=True):
    """Raffinamento del passo temporale su una mesh fissa. dts: lista di passi (rapporto 2 consigliato)."""
    righe = []
    ctx = Contesto(geo, n, p, r_filtro=r_filtro)
    for dt in dts:
        pp = dict(p); pp['dt'] = float(dt)
        serie, info = simula_diag(ctx, pp, T_target=T_target, durata_max=durata_max)
        val, stato = valori_a_tau(serie, T_target)
        val.update(dt=dt, stato=info['stato'], chiusura_max=float(serie['chiusura_energia'].abs().max()))
        righe.append(val)
        if verbose:
            print(f"  dt={dt:g} s: stato={info['stato']}, tau*={val.get('tau_star', np.nan):.1f} s")
    return pd.DataFrame(righe)


def studio_interfaccia(geo, griglie, p, w_fissa_mm=None, verbose=True):
    """Confronta, sul solo problema elettrico, (a) rampa w ~ h, (b) rampa fissa, (c) rampa ~ h con correzione del volume."""
    out = {}
    out['prop'], h = studio_elettrico(geo, griglie, p, 'prop', verbose=False)
    out['prop_corr_volume'], _ = studio_elettrico(geo, griglie, p, 'prop', corr_volume=True, verbose=False)
    if w_fissa_mm:
        out['fissa'], _ = studio_elettrico(geo, griglie, p, 'fissa', w_fissa=w_fissa_mm / 1000.0, verbose=False)
    return out, h


# ---------------------------------------------------------------------------------------------------------
# 7. Piano fattoriale 2^k con punto centrale
# ---------------------------------------------------------------------------------------------------------
def applica_fattori(p_base, valori):
    """Traduce i valori dei fattori in parametri del motore.
    Fattori riconosciuti: sig_m20, rapporto_sig (sigma_p20/sigma_m20), campo_V_cm, T0, fattore_h."""
    p = dict(p_base)
    s_m = valori.get('sig_m20', p_base['_sig_m20'])
    alfa_m = p_base['_alfa_m']
    rapporto = valori.get('rapporto_sig', p_base['_sig_p20'] / p_base['_sig_m20'])
    s_p = rapporto * s_m
    alfa_p = p_base['_alfa_p']
    p['matrice_a'], p['matrice_b'] = s_m * alfa_m, s_m * (1 - 20 * alfa_m)
    p['polpetta_a'], p['polpetta_b'] = s_p * alfa_p, s_p * (1 - 20 * alfa_p)
    if 'campo_V_cm' in valori:
        p['V_rms'] = valori['campo_V_cm'] * p['_Lx_cm']
    if 'T0' in valori:
        p['T_iniziale'] = p['T_iniziale_polpetta'] = valori['T0']
    if 'fattore_h' in valori:
        p['fattore_h'] = valori['fattore_h']
    return p


def piano_fattoriale(geo, p_base, fattori, n_mesh, T_target=T_COTTURA_C, uscite=None, r_filtro=3e-3, durata_max=1500.0, verbose=True):
    """fattori: dict nome -> (livello_basso, livello_alto). Esegue le 2^k prove + 1 punto centrale; restituisce (prove, effetti).
    Uscite a tau*. Casi con stato diverso da 'ok' (tau non raggiunto o ebollizione prima di tau*) hanno uscite NaN e vengono segnalati."""
    uscite = uscite or ['tau_star', 'scarto_pol_pur', 'T_max_meno_T_min', 'margine_ebollizione', 'frac_P_p', 'E_pm', 'frac_E_perdite', 'E_in']
    nomi = list(fattori)
    ctx = Contesto(geo, n_mesh, p_base, r_filtro=r_filtro)
    prove = []
    schema = list(itertools.product([-1, 1], repeat=len(nomi))) + [tuple([0] * len(nomi))]
    for k, cod in enumerate(schema):
        vals = {nm: (0.5 * (fattori[nm][0] + fattori[nm][1]) if c == 0 else fattori[nm][0 if c < 0 else 1]) for nm, c in zip(nomi, cod)}
        p = applica_fattori(p_base, vals)
        # il contesto dipende dai parametri solo per tabelle termiche/entalpia (indipendenti dai fattori elettrici/T0/h): si riusa
        serie, info = simula_diag(ctx, p, T_target=T_target, durata_max=durata_max)
        val, stato = valori_a_tau(serie, T_target)
        riga = dict(prova=k, **{f'X_{nm}': c for nm, c in zip(nomi, cod)}, **vals, stato=info['stato'])
        for u in uscite:
            riga[u] = val.get(u, np.nan)
        prove.append(riga)
        if verbose:
            print(f"  prova {k + 1}/{len(schema)} {dict(zip(nomi, cod))}: stato={info['stato']}, tau*={riga.get('tau_star', np.nan):.0f} s")
    prove = pd.DataFrame(prove)
    return prove, effetti_fattoriali(prove, nomi, uscite)


def effetti_fattoriali(prove, nomi, uscite):
    """Effetti principali e interazioni (fino all'ordine k) su variabili codificate +-1.
    Effetto del contrasto S = (media delle prove con prodotto codificato +1) - (media con -1). Curvatura = media vertici - punto centrale.
    Simulazioni deterministiche: i valori p della regressione NON hanno il significato usuale; si interpreta la dimensione degli effetti."""
    P = prove[prove[[f'X_{n}' for n in nomi]].abs().sum(axis=1) > 0]
    C = prove[prove[[f'X_{n}' for n in nomi]].abs().sum(axis=1) == 0]
    ok = P['stato'] == 'tau_raggiunto'
    righe = []
    for u in uscite:
        y = P[u].values
        for ordine in range(1, len(nomi) + 1):
            for comb in itertools.combinations(nomi, ordine):
                prod = np.prod([P[f'X_{n}'].values for n in comb], axis=0)
                m = ok.values & np.isfinite(y)
                if m.sum() < 2 or len(np.unique(prod[m])) < 2:
                    eff = np.nan
                else:
                    eff = y[m & (prod > 0)].mean() - y[m & (prod < 0)].mean()
                righe.append(dict(uscita=u, termine=':'.join(comb), ordine=ordine, effetto=eff))
        curv = (np.nanmean(y[ok.values]) - float(C[u].iloc[0])) if len(C) and np.isfinite(C[u].iloc[0]) else np.nan
        righe.append(dict(uscita=u, termine='curvatura', ordine=0, effetto=curv))
    ef = pd.DataFrame(righe)
    ef['n_prove_valide'] = int(ok.sum())
    ef['n_prove_totali'] = int(len(P))
    return ef


# ---------------------------------------------------------------------------------------------------------
# 8. Riferimento a interfaccia netta con mesh conforme (opzionale; richiede gmsh e meshio)
# ---------------------------------------------------------------------------------------------------------
def _mesh_conforme_gmsh_impl(geo, lc_globale, lc_interfaccia, file_msh='conforme.msh'):
    """Costruisce con gmsh un parallelepipedo con sfere conformi (gruppi fisici 'purea' e 'polpette'). Dimensioni in metri."""
    import gmsh
    gmsh.initialize(['-v', '0'], interruptible=False)
    gmsh.option.setNumber('General.Terminal', 0); gmsh.option.setNumber('General.Verbosity', 0)      # nessun messaggio sul terminale
    gmsh.model.add('cella')
    box = gmsh.model.occ.addBox(0, 0, 0, geo['Lx'], geo['Ly'], geo['Lz'])
    sfere = [gmsh.model.occ.addSphere(c[0], c[1], c[2], geo['R']) for c in geo['centri']]
    if sfere:
        out, mappa = gmsh.model.occ.fragment([(3, box)], [(3, s) for s in sfere])
        gmsh.model.occ.synchronize()
        vols = [v[1] for v in gmsh.model.getEntities(3)]
        pol = []
        for v in vols:                                   # una sfera ha un riquadro di ingombro di lato ~ 2R; la purea occupa tutta la cella
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(3, v)
            if max(x1 - x0, y1 - y0, z1 - z0) < 2.1 * geo['R']:
                pol.append(v)
        pur = [v for v in vols if v not in pol]
        gmsh.model.addPhysicalGroup(3, pur, 1, 'purea')
        gmsh.model.addPhysicalGroup(3, pol, 2, 'polpette')
    else:
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [box], 1, 'purea')
    # dimensione elemento: fine vicino alle superfici delle sfere
    gmsh.option.setNumber('Mesh.MeshSizeMax', lc_globale)
    gmsh.option.setNumber('Mesh.MeshSizeMin', lc_interfaccia)
    pts = gmsh.model.getEntities(0)
    gmsh.model.mesh.setSize(pts, lc_globale)
    if geo['centri']:
        sup = []
        for e in gmsh.model.getEntities(2):              # superfici delle sfere: riquadro di ingombro <= 2R
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, e[1])
            if max(x1 - x0, y1 - y0, z1 - z0) < 2.1 * geo['R']:
                sup.append(e[1])
        f = gmsh.model.mesh.field.add('Distance'); gmsh.model.mesh.field.setNumbers(f, 'SurfacesList', sup)
        t = gmsh.model.mesh.field.add('Threshold'); gmsh.model.mesh.field.setNumber(t, 'InField', f)
        gmsh.model.mesh.field.setNumber(t, 'SizeMin', lc_interfaccia); gmsh.model.mesh.field.setNumber(t, 'SizeMax', lc_globale)
        gmsh.model.mesh.field.setNumber(t, 'DistMin', 0.5 * lc_interfaccia); gmsh.model.mesh.field.setNumber(t, 'DistMax', 2.0 * geo['R'])
        gmsh.model.mesh.field.setAsBackgroundMesh(t)
    gmsh.model.mesh.generate(3)
    gmsh.write(file_msh)
    gmsh.finalize()
    return file_msh


import threading as _threading
_GMSH_LOCK = _threading.RLock()


def mesh_conforme_gmsh(*a, **k):
    """Generazione della mesh con gmsh, serializzata (gmsh non e' thread-safe) e senza gestione dei segnali (utilizzabile fuori dal thread principale)."""
    with _GMSH_LOCK:
        try:
            return _mesh_conforme_gmsh_impl(*a, **k)
        finally:
            try:
                import gmsh
                if gmsh.isInitialized(): gmsh.finalize()
            except Exception:
                pass


def elettrico_conforme(geo, p, file_msh):
    """Problema elettrico (T uniforme) su mesh conforme: sigma costante per elemento (purea/polpette)."""
    import meshio
    from skfem import Mesh
    m = meshio.read(file_msh)
    mesh = MeshTet(m.points.T, m.cells_dict['tetra'].T.astype(np.int64))
    tags = m.cell_data_dict['gmsh:physical']['tetra']
    basis = Basis(mesh, ElementTetP1())
    dx = basis.dx
    T = p['T_iniziale']
    s_m = float(p['matrice_a'] * T + p['matrice_b'])
    s_p = float(p['polpetta_a'] * p.get('T_iniziale_polpetta', T) + p['polpetta_b'])
    es = np.where(tags == 2, s_p, s_m)[:, None] * np.ones_like(dx)

    @BilinearForm
    def laplace(u, v, w):
        return w['sigma'] * dot(grad(u), grad(v))
    dofs_sx = basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    dofs_dx = basis.get_dofs(lambda x: np.isclose(x[0], geo['Lx'])).all()
    from skfem import condense, solve
    A = laplace.assemble(basis, sigma=es)
    V0 = 1.0 - basis.doflocs[0] / geo['Lx']
    D = np.concatenate([dofs_sx, dofs_dx])
    Vu = solve(*condense(A, basis.zeros(), x=V0, D=D))
    gV = basis.interpolate(Vu).grad
    V_rms = p['V_rms']
    q = V_rms ** 2 * es * (gV[0] ** 2 + gV[1] ** 2 + gV[2] ** 2)
    res = A @ Vu
    phi = (tags == 2).astype(float)[:, None] * np.ones_like(dx)
    P = float((q * dx).sum()); P_p = float((q * dx * phi).sum())
    Xq = basis.global_coordinates().value
    reg = regioni_geometriche(geo, Xq, phi, 0.5)
    out = dict(P=P, I_L=V_rms * float(res[dofs_sx].sum()), frac_P_p=P_p / P, n_nodi=int(basis.N), h_min_mm=float(np.min(mesh.param())) * 1000)
    for nome, w in reg.items():
        ww = w * dx
        out['q_' + nome] = float((q * ww).sum() / ww.sum())
    out['q_media_polp'] = float((q * dx * phi).sum() / (dx * phi).sum())
    out['q_media_pur'] = float((q * dx * (1 - phi)).sum() / (dx * (1 - phi)).sum())
    out['q_p01'], out['q_p99'] = [float(v) for v in percentile_pesato(q.ravel(), dx.ravel(), [1, 99])]
    return out


# ---------------------------------------------------------------------------------------------------------
# 9. Criterio di accettazione e grafici
# ---------------------------------------------------------------------------------------------------------
def griglia_minima(valori, h, tab_conv, soglia_percento, grandezze):
    """Griglia piu' grossolana per cui, per tutte le grandezze indicate, lo scarto relativo dal valore estrapolato (Richardson,
    procedura di Celik) e' <= soglia (in %), e lo stesso vale per tutte le griglie piu' fini. Restituisce (indice, tabella degli scarti).
    NOTA: la soglia e' una scelta di progetto (da legare all'incertezza di misura u_D secondo ASME V&V 20), non un valore di letteratura."""
    h = np.asarray(h, float)
    ordine = np.argsort(h)                               # dalla piu' fine
    V = valori.iloc[ordine].reset_index(drop=True)
    sc = {}
    for g in grandezze:
        r = tab_conv[tab_conv.grandezza == g]
        if r.empty or not np.isfinite(r.f_ext.iloc[0]) or r.f_ext.iloc[0] == 0:
            continue
        sc[g] = 100 * np.abs(V[g].values - r.f_ext.iloc[0]) / abs(r.f_ext.iloc[0])
    tab = pd.DataFrame(sc)
    tab.insert(0, 'h_rep_mm', 1000 * h[ordine])
    ok = (tab[list(sc)] <= soglia_percento).all(axis=1).values
    idx = None
    for i in range(len(ok)):                              # dalla piu' fine verso la piu' grossolana
        if ok[:i + 1].all():
            idx = i
    return idx, tab


def grafico_convergenza(h, valori, tab_conv, grandezze, titolo='', colonne=2, file=None):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13, 'legend.fontsize': 11})
    grandezze = [g for g in grandezze if g in valori.columns]
    righe = int(np.ceil(len(grandezze) / colonne))
    fig, axs = plt.subplots(righe, colonne, figsize=(6.6 * colonne, 4.4 * righe), squeeze=False)
    h = np.asarray(h, float) * 1000
    for ax, g in zip(axs.ravel(), grandezze):
        y = valori[g].values
        ax.plot(h, y, 'o-', color='tab:blue', lw=2, ms=7, label='calcolato')
        r = tab_conv[tab_conv.grandezza == g]
        if not r.empty and np.isfinite(r.f_ext.iloc[0]):
            fe, gci = r.f_ext.iloc[0], r.GCI_fine_percento.iloc[0]
            ax.axhline(fe, color='tab:red', ls='--', lw=1.8, label='estrapolato')
            f1 = y[np.argmin(h)]
            if np.isfinite(gci):
                ax.fill_between([h.min() * 0.9, h.max() * 1.1], f1 * (1 - gci / 100), f1 * (1 + gci / 100), color='tab:orange', alpha=0.25, label='banda GCI (fine)')
            ax.set_title(f"{g}\np={r.p_osservato.iloc[0]:.2f}, {r.regime.iloc[0]}, GCI={gci:.2f}%")
        else:
            ax.set_title(g)
        ax.set_xscale('log'); ax.set_xlabel('h rappresentativo [mm]'); ax.grid(alpha=0.3, which='both')
        ax.legend(loc='best')
    for ax in axs.ravel()[len(grandezze):]:
        ax.axis('off')
    if titolo:
        fig.suptitle(titolo, fontsize=15)
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=150, bbox_inches='tight')
    return fig


def grafico_effetti(effetti, uscita, file=None, max_termini=12):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 13})
    e = effetti[(effetti.uscita == uscita) & (effetti.termine != 'curvatura')].copy()
    e['abs'] = e.effetto.abs()
    e = e.sort_values('abs', ascending=True).tail(max_termini)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(e) + 2.2))
    ax.barh(e.termine, e.effetto, color=['tab:red' if v < 0 else 'tab:blue' for v in e.effetto])
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel(f'effetto su {uscita}'); ax.set_title(f'Effetti (variabili codificate -1/+1): {uscita}')
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=150, bbox_inches='tight')
    return fig


# =========================================================================================================
# 10. ITERAZIONE SU TUTTE LE CONFIGURAZIONI E LE RICETTE  (rev. 2)
# =========================================================================================================
import json
import os
import traceback

CONFIGURAZIONI = {                      # posizioni P1, P2, P3 occupate da una polpetta (Middle e Side dedotte dai dati: da confermare)
    'Tris': (True, True, True), 'Symm': (True, False, True), 'Asymm': (True, True, False),
    'Middle': (False, True, False), 'Side': (True, False, False), 'RP': (False, False, False),
}
PARAMETRI_CAL = ['sig_m20', 'alfa_m', 'sig_p20', 'alfa_p', 'fk_m', 'fk_p', 'fc_m', 'fc_p', 'fh']
NOMI_PARAM = {'sig_m20': 'sigma20 purea', 'alfa_m': 'alfa purea', 'sig_p20': 'sigma20 polpette', 'alfa_p': 'alfa polpette',
              'fk_m': 'k purea (fattore)', 'fk_p': 'k polpette (fattore)', 'fc_m': 'rho*cp purea (fattore)', 'fc_p': 'rho*cp polpette (fattore)',
              'fh': 'h scambio (fattore)'}
CELLA_UNISA = 'UNISA (54 x 99 x 66 mm)'
TENSIONI = (30, 40, 50)


def registro_ricette():
    """Ricette sperimentali A-E (sale 2.9, 5.7, 0, 0.725 e 1.45%) a 30, 40 e 50 V: chiave 'A_40V'. Composizione dalle masse della tabella delle ricette (% w/w sulla massa totale),
    sigma(T) della purea dalla regressione ricavata dal grafico delle conducibilita' (me.RICETTE_SPERIMENTALI). La tensione e' un attributo della ricetta."""
    R = {}
    for nome, r in me.RICETTE_SPERIMENTALI.items():
        m = r['masse']; tot = sum(m.values()); a, b = r['sigma'][:2]; s20 = 20 * a + b
        comp = me.composizione_purea(100 * m['burro'] / tot, 100 * m['sale'] / tot, 100 * m['fiocchi'] / tot, 100 * m['acqua'] / tot)
        for V in TENSIONI:
            R[f'{nome}_{V}V'] = dict(cella=CELLA_UNISA, comp=comp, sigma_m20=s20, alfa_m=a / s20, V=float(V), sale=r['sale_pc'], origine=f"{r['sigma'][2]} ({r['sigma'][3]})")
    return R


CONFIG_TUTTE = ('Tris', 'Symm', 'Asymm', 'Middle', 'Side')


def piano_completo():
    """Tutti i casi: ogni ricetta (A-E), ogni configurazione e ogni tensione (30, 40 e 50 V): 5 x 5 x 3 = 75 casi. Chiave del caso: ('A_40V', 'Tris')."""
    return [(f'{r}_{V}V', c) for r in sorted(me.RICETTE_SPERIMENTALI) for c in CONFIG_TUTTE for V in TENSIONI]


def costruisci_caso(nome_ricetta, nome_config, ricette=None, campo_V_cm=4.0, T0=20.0, T_amb=21.0, dt=10.0, durata=1500.0,
                    a_polp=None, b_polp=None):
    """Costruisce geometria e parametri del motore per una coppia (ricetta, configurazione)."""
    ricette = ricette or registro_ricette()
    r = ricette[nome_ricetta]
    geo = me.geometria(r['cella'], 'RP (solo purea)', dict(occupazione=list(CONFIGURAZIONI[nome_config])))
    a_p = me.SIGMA_POLPETTE[0] if a_polp is None else a_polp
    b_p = me.SIGMA_POLPETTE[1] if b_polp is None else b_polp
    s_p20 = 20 * a_p + b_p
    p = dict(me.PARAMETRI_DEFAULT)
    p.update(comp_matrice=r['comp'], matrice_a=r['sigma_m20'] * r['alfa_m'], matrice_b=r['sigma_m20'] * (1 - 20 * r['alfa_m']),
             polpetta_a=a_p, polpetta_b=b_p, V_rms=r['V'] if 'V' in r else campo_V_cm * geo['Lx'] * 100, T_iniziale=T0, T_iniziale_polpetta=T0, T_amb=T_amb,
             h_modo='naturale_irraggiamento', prop_modo='choi_okos_T', dt=dt, durata=durata,
             _sig_m20=r['sigma_m20'], _alfa_m=r['alfa_m'], _sig_p20=s_p20, _alfa_p=a_p / s_p20, _Lx_cm=geo['Lx'] * 100)
    return dict(id=f'{nome_ricetta}__{nome_config}', ricetta=nome_ricetta, config=nome_config, cella=r['cella'], geo=geo, p=p)


def griglie_per_geo(geo, h_base_mm=10.0, livelli=4, max_nodi=150000):
    """Serie a rapporto 2: intervalli n_i = round(L_i/h_base) * 2^k. I livelli oltre max_nodi vengono scartati."""
    base = [max(2, int(round(L * 1000.0 / h_base_mm))) for L in (geo['Lx'], geo['Ly'], geo['Lz'])]
    out = []
    for k in range(livelli):
        n = tuple(b * 2 ** k for b in base)
        if (n[0] + 1) * (n[1] + 1) * (n[2] + 1) <= max_nodi:
            out.append(n)
    return out


def valori_a_tempo(serie, t_eval):
    """Interpolazione lineare nel tempo di tutte le colonne numeriche; NaN se t_eval e' fuori dal tempo simulato."""
    if t_eval > serie['t'].max() + 1e-9:
        return {}
    out = {}
    for c in serie.columns:
        if c == 't':
            continue
        out[c] = float(np.interp(t_eval, serie['t'].values, serie[c].values))
    out['t'] = float(t_eval)
    out['margine_ebollizione'] = T_BOLLITURA_C - out['T_max']
    out['scarto_pol_pur'] = out['T_media_pur'] - out['T_media_polp']
    return out


def _riferimento(f, h, tab_conv, g):
    """Valore di riferimento per lo scarto: estrapolato se la convergenza e' monotona con ordine credibile, altrimenti griglia fine."""
    r = tab_conv[tab_conv.grandezza == g]
    f_fine = f[np.argmin(h)]
    if not r.empty and r.regime.iloc[0] == 'monotona' and 0.5 <= r.p_osservato.iloc[0] <= 2.5 and np.isfinite(r.f_ext.iloc[0]):
        return float(r.f_ext.iloc[0]), 'estrapolato'
    return float(f_fine), 'griglia fine'


def mesh_minima(valori, h, tab_conv, soglia_percento, grandezze, scala=None, tol_abs=None):
    """Indice (0 = piu' fine, in ordine di h crescente) della griglia piu' grossolana per cui tutte le grandezze hanno scarto accettabile
    rispetto al riferimento, e lo stesso vale per tutte le griglie piu' fini. Uno scarto e' accettabile se e' <= soglia_percento (rispetto a
    'scala[g]' se data, altrimenti al valore di riferimento) OPPURE se il suo valore assoluto e' <= tol_abs[g] (es. 0.2 C per le temperature).
    Restituisce (indice o None, dettaglio per grandezza con la dimensione minima richiesta da ciascuna)."""
    h = np.asarray(h, float)
    ordine = np.argsort(h)
    hs = h[ordine]
    V = valori.iloc[ordine].reset_index(drop=True)
    righe, OK = [], []
    for g in grandezze:
        if g not in V.columns or not np.all(np.isfinite(V[g].values)):
            continue
        f = V[g].values.astype(float)
        ref, tipo = _riferimento(valori[g].values.astype(float), h, tab_conv, g)
        den = max(abs(scala[g]) if (scala and g in scala) else abs(ref), 1e-12)
        sa = np.abs(f - ref)
        sc = 100 * sa / den
        ok = sc <= soglia_percento
        if tol_abs and g in tol_abs:
            ok = ok | (sa <= tol_abs[g])
        i_g = None
        for i in range(len(ok)):
            if ok[:i + 1].all():
                i_g = i
        riga = dict(grandezza=g, riferimento=tipo, scarto_fine_pc=float(sc[0]), scarto_abs_fine=float(sa[0]),
                    h_min_richiesta_mm=np.nan if i_g is None else float(hs[i_g] * 1000))
        riga.update({f'scarto_h{hs[i] * 1000:.2f}mm': float(sc[i]) for i in range(len(f))})
        righe.append(riga); OK.append(ok)
    det = pd.DataFrame(righe)
    if det.empty:
        return None, det
    allok = np.array(OK).all(axis=0)
    idx = None
    for i in range(len(allok)):
        if allok[:i + 1].all():
            idx = i
    return idx, det


def mesh_minima_successive(valori, h, grandezze, soglia_percento, scala=None, tol_abs=None):
    """Criterio per DIFFERENZE SUCCESSIVE (livello 2): la griglia k (0 = piu' fine) e' accettabile se la variazione di ogni grandezza
    passando da k alla griglia piu' fine successiva e' <= soglia (percentuale oppure assoluta) e lo stesso vale per tutte le coppie piu' fini.
    Non usa l'estrapolazione, quindi non dipende da un ordine di convergenza credibile. Restituisce l'indice (>= 1) o None."""
    h = np.asarray(h, float)
    V = valori.iloc[np.argsort(h)].reset_index(drop=True)
    OK = []
    for g in grandezze:
        if g not in V.columns or not np.all(np.isfinite(V[g].values)) or len(V) < 2:
            continue
        f = V[g].values.astype(float)
        d = np.abs(np.diff(f))
        den = np.array([max(abs(scala[g]) if (scala and g in scala) else abs(f[k - 1]), 1e-12) for k in range(1, len(f))])
        ok = (100 * d / den) <= soglia_percento
        if tol_abs and g in tol_abs:
            ok = ok | (d <= tol_abs[g])
        OK.append(ok)
    if not OK:
        return None
    allok = np.array(OK).all(axis=0)
    idx = None
    for j in range(len(allok)):
        if allok[:j + 1].all():
            idx = j + 1
    return idx


def dt_minimo(valori, dts, tab_conv, soglia_percento, grandezze, scala=None, tol_abs=None):
    """Passo temporale piu' grande per cui lo scarto dal riferimento e' <= soglia (e lo stesso vale per i passi piu' piccoli)."""
    dts = np.asarray(dts, float)
    idx, det = mesh_minima(valori, dts, tab_conv, soglia_percento, grandezze, scala, tol_abs)
    return (None if idx is None else float(np.sort(dts)[idx])), det


_QOI_T = ('T_max', 'T_media_polp', 'T_media_pur', 'T_P1', 'T_P2', 'T_P3', 'scarto_pol_pur')


def valuta_convergenza(valori, h, ess, loc, opz, scala=None, tol_abs=None):
    """Valutazione completa di uno studio (mesh o passo temporale) per le grandezze essenziali e locali.
    Restituisce: tabella di convergenza, indici e dimensioni minime (essenziali; essenziali + locali), dimensione piu' fine provata,
    GRANDEZZA CHE BLOCCA la soglia (se nessuna griglia la soddisfa: quelle che falliscono sulla griglia fine; altrimenti la piu' esigente)."""
    tol = tol_abs if tol_abs is not None else {g: opz.get('tol_abs_C', 0.2) for g in _QOI_T}
    out = dict(tab=pd.DataFrame(), idx_ess=None, idx_loc=None, h_ess=None, h_loc=None, h_fine=float(np.min(h)) * 1000 if len(h) else np.nan,
               bloccante='', scarto_bloccante=np.nan, GCI_ess_max=np.nan, det=pd.DataFrame())
    gr = [g for g in list(ess) + list(loc) if g in valori.columns and np.all(np.isfinite(valori[g].values))]
    if len(valori) < 3 or not gr:
        return out
    tab = tabella_convergenza(h, valori, gr)
    e_ok = [g for g in ess if g in gr]
    l_ok = [g for g in loc if g in gr]
    ie, de = mesh_minima(valori, h, tab, opz['soglia'], e_ok, scala, tol)
    il, _ = mesh_minima(valori, h, tab, opz['soglia'], e_ok + l_ok, scala, tol)
    hs = np.sort(np.asarray(h, float))
    out.update(tab=tab, idx_ess=ie, idx_loc=il, h_ess=None if ie is None else float(hs[ie] * 1000),
               h_loc=None if il is None else float(hs[il] * 1000), det=de)
    if not de.empty:
        if ie is None:
            b = de[de.h_min_richiesta_mm.isna()].sort_values('scarto_fine_pc', ascending=False)
            if len(b):
                out['bloccante'] = ','.join(b.grandezza.tolist()); out['scarto_bloccante'] = float(b.scarto_fine_pc.iloc[0])
        else:
            b = de.sort_values('h_min_richiesta_mm')
            out['bloccante'] = str(b.grandezza.iloc[0]); out['scarto_bloccante'] = float(b.scarto_fine_pc.iloc[0])
    gg = []
    for _, rr in tab[tab.grandezza.isin(e_ok)].iterrows():
        if scala and rr.grandezza in scala and scala[rr.grandezza]:
            gg.append(rr.GCI_fine_percento / 100.0 * abs(rr.f_fine) / abs(scala[rr.grandezza]) * 100.0)     # % dell'incremento di temperatura
        else:
            gg.append(rr.GCI_fine_percento)
    gg = np.array(gg, float)
    out['GCI_ess_max'] = float(np.nanmax(gg)) if len(gg) else np.nan
    # ---- raccomandazione che esiste SEMPRE: livello 1 (GCI vs estrapolato), 2 (differenze successive), 3 (griglia piu' fine provata) ----
    if ie is not None:
        rec, liv = ie, 1
    else:
        isu = mesh_minima_successive(valori, h, e_ok, opz['soglia'], scala, tol)
        rec, liv = (isu, 2) if isu is not None else (0, 3)
    out.update(idx_rec=int(rec), livello=int(liv), h_rec=float(hs[rec] * 1000))
    # ---- dimensione stimata necessaria per la soglia (estrapolazione della legge di convergenza osservata) ----
    out.update(h_stimata=np.nan, nodi_stimati=np.nan)
    if ie is None and not de.empty:
        req = []
        for _, r in de.iterrows():
            if pd.isna(r.h_min_richiesta_mm) and r.scarto_fine_pc > opz['soglia'] and r.scarto_fine_pc > 0:
                tp = tab[tab.grandezza == r.grandezza]
                p_ = float(np.clip(tp.p_osservato.iloc[0], 0.5, 2.0)) if (len(tp) and np.isfinite(tp.p_osservato.iloc[0])) else 1.0
                req.append(hs[0] * (opz['soglia'] / r.scarto_fine_pc) ** (1.0 / p_) * 1000)
        if req:
            out['h_stimata'] = float(min(req))
            if 'n_nodi' in valori.columns:
                nf = float(valori['n_nodi'].values[np.argmin(h)])
                out['nodi_stimati'] = nf * (hs[0] * 1000 / out['h_stimata']) ** 3
    # ---- mesh applicabile nell'applicazione (tetto ai nodi) e scarto massimo che le corrisponde ----
    out.update(h_app=np.nan, nodi_app=np.nan, err_app=np.nan, grand_app='')
    if 'n_nodi' in valori.columns and not de.empty:
        nn = valori['n_nodi'].values[np.argsort(h)]
        lim = opz.get('max_nodi_app', 35000)
        cand = [i for i in range(len(nn)) if nn[i] <= lim]
        if cand:
            ia = cand[0]
            cols = [c for c in de.columns if c.startswith('scarto_h')]
            if ia < len(cols):
                v = de[cols[ia]].values
                out.update(h_app=float(hs[ia] * 1000), nodi_app=float(nn[ia]), err_app=float(np.nanmax(v)), grand_app=str(de.grandezza.iloc[int(np.nanargmax(v))]))
    return out


# ---------------------------------------------------------------------------------------------------------
# Sensitivita' locale scalata e identificabilita' (scelta dei parametri da calibrare)
# ---------------------------------------------------------------------------------------------------------
def applica_param(p_base, theta):
    p = dict(p_base)
    s_m, al_m, s_p, al_p = theta['sig_m20'], theta['alfa_m'], theta['sig_p20'], theta['alfa_p']
    p['matrice_a'], p['matrice_b'] = s_m * al_m, s_m * (1 - 20 * al_m)
    p['polpetta_a'], p['polpetta_b'] = s_p * al_p, s_p * (1 - 20 * al_p)
    p['fk_matrice'], p['fk_polpetta'], p['fc_matrice'], p['fc_polpetta'], p['fattore_h'] = theta['fk_m'], theta['fk_p'], theta['fc_m'], theta['fc_p'], theta['fh']
    return p


def sensibilita_locale(geo, p_base, n_mesh, dt, durata=270.0, delta=0.10, passo_misura=5.0, sigma_misura=0.5, se_max=0.20, sonde_fn=None):
    """Matrice di sensitivita' scalata X_ij = theta_j dT_i/dtheta_j (differenze centrali relative +-delta) sulle temperature alle sonde a
    tempi di misura equispaziati, con le funzioni dell'applicazione (me.simula). Restituisce (tabella dei parametri, matrici, selezione).
    sigma_misura: deviazione standard del rumore di misura [C] (PROVVISORIA: non nota); se_max: errore standard relativo massimo ammesso."""
    if sonde_fn is None:
        mesh, basis, h_eff, _ = costruisci_mesh_n(geo, n_mesh)
    theta0 = dict(sig_m20=p_base['_sig_m20'], alfa_m=p_base['_alfa_m'], sig_p20=p_base['_sig_p20'], alfa_p=p_base['_alfa_p'],
                  fk_m=1.0, fk_p=1.0, fc_m=1.0, fc_p=1.0, fh=1.0)
    pb = dict(p_base); pb['dt'] = float(dt); pb['durata'] = float(durata)
    tempi = np.arange(passo_misura, durata + 1e-9, passo_misura)

    def sonde(theta):
        if sonde_fn is not None:                  # sonde_fn(theta, pb, tempi) -> (matrice sonde x tempi, fase alle sonde): es. mesh conformi
            return sonde_fn(theta, pb, tempi)
        ris = me.simula(geo, mesh, basis, h_eff, applica_param(pb, theta))
        return np.vstack([np.interp(tempi, ris['tempo'], ris['T_sonde'][n]) for n in sorted(ris['T_sonde'])]), ris['fase_sonda']
    T0m, fase = sonde(theta0)
    nomi_s = sorted(fase)
    cols = {}
    for j in PARAMETRI_CAL:
        tp, tm = dict(theta0), dict(theta0)
        tp[j] = theta0[j] * (1 + delta); tm[j] = theta0[j] * (1 - delta)
        cols[j] = ((sonde(tp)[0] - sonde(tm)[0]) / (2 * delta)).ravel()          # [C] per variazione relativa unitaria
    X = np.column_stack([cols[j] for j in PARAMETRI_CAL])
    in_pol = np.repeat([fase[n] > 0.5 for n in nomi_s], len(tempi))
    sottoinsiemi = {'tutte': np.ones(len(in_pol), bool)}
    if (~in_pol).any():
        sottoinsiemi['purea'] = ~in_pol
    if in_pol.any():
        sottoinsiemi['polpetta'] = in_pol
    righe, selez = [], {}
    for nome, m in sottoinsiemi.items():
        Xs = X[m]
        norme = np.sqrt((Xs ** 2).mean(axis=0))
        # correlazione tra colonne (coseno): quanto due parametri sono confondibili
        nn = np.linalg.norm(Xs, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            C = (Xs.T @ Xs) / np.outer(nn, nn)
        try:
            cov = sigma_misura ** 2 * np.linalg.pinv(Xs.T @ Xs)
            se = np.sqrt(np.abs(np.diag(cov)))
        except Exception:
            se = np.full(len(PARAMETRI_CAL), np.nan)
        sv = np.linalg.svd(Xs, compute_uv=False)
        # selezione greedy: parametri in ordine di sensitivita', aggiunti se tutti i selezionati restano entro se_max
        sel = []
        for j in np.argsort(-norme):
            if norme[j] <= 0 or not np.isfinite(norme[j]):
                continue
            prova = sel + [j]
            Xt = Xs[:, prova]
            try:
                se_t = sigma_misura * np.sqrt(np.abs(np.diag(np.linalg.inv(Xt.T @ Xt))))
            except np.linalg.LinAlgError:
                continue
            if np.all(se_t <= se_max):
                sel = prova
        selez[nome] = [PARAMETRI_CAL[j] for j in sel]
        Cabs = np.abs(np.nan_to_num(C)); np.fill_diagonal(Cabs, 0.0)
        for k, j in enumerate(PARAMETRI_CAL):
            kk = int(np.argmax(Cabs[k]))
            righe.append(dict(sonde=nome, parametro=j, effetto_10pc_C=0.1 * norme[k], se_relativo=se[k],
                              se_da_solo=float(sigma_misura / max(norme[k], 1e-12)),
                              identificabile_da_solo=bool(sigma_misura / max(norme[k], 1e-12) <= se_max), selezionato=bool(k in sel),
                              corr_max_con=PARAMETRI_CAL[kk], corr_max=float(Cabs[k, kk]), cond=float(sv[0] / max(sv[-1], 1e-300))))
    return pd.DataFrame(righe), X, selez


# ---------------------------------------------------------------------------------------------------------
# Analisi completa di un caso
# ---------------------------------------------------------------------------------------------------------
GRAND_ESS_A = ['P', 'frac_P_p', 'q_media_polp', 'q_media_pur']
GRAND_LOC_A = ['q_strati_z', 'q_asse', 'q_p99']
GRAND_ESS_B = ['T_media_polp', 'T_media_pur', 'scarto_pol_pur', 'T_P1', 'T_P2', 'T_P3', 'E_in']
GRAND_ESS_B_TAU = ['tau_star', 'scarto_pol_pur', 'T_max', 'E_pm']
GRAND_LOC_B = ['T_max', 'q_asse', 'q_p99']


def _aggiungi_raccomandazione(riep, vA, vB=None, vC=None, vAc=None):
    """Imposta nel riepilogo la raccomandazione che esiste SEMPRE (h_raccomandata_mm, livello, h stimata necessaria, mesh applicabile nell'app)."""
    def put(pref, v):
        if v is None or 'h_rec' not in v:
            return
        riep.update({f'{pref}_h_rec_mm': v['h_rec'], f'{pref}_livello': v['livello'], f'{pref}_h_stimata_mm': v['h_stimata'], f'{pref}_nodi_stimati': v['nodi_stimati'],
                     f'{pref}_h_app_mm': v['h_app'], f'{pref}_nodi_app': v['nodi_app'], f'{pref}_err_app': v['err_app'], f'{pref}_grand_app': v['grand_app']})
    put('A', vA); put('B', vB)
    if vC is not None and 'h_rec' in vC:
        riep.update(C_dt_rec=vC['h_rec'], C_livello=vC['livello'])
    if vAc is not None and 'h_rec' in vAc:
        riep.update(Acorr_h_rec_mm=vAc['h_rec'], Acorr_livello=vAc['livello'], Acorr_h_stimata_mm=vAc['h_stimata'], Acorr_GCI_ess_max=vAc['GCI_ess_max'],
                    Acorr_bloccante=vAc['bloccante'], Acorr_err_app=vAc['err_app'], Acorr_h_ess_mm=vAc['h_ess'])
    hs = [riep.get(k) for k in ('A_h_rec_mm', 'B_h_rec_mm') if riep.get(k) is not None and np.isfinite(riep.get(k))]
    riep['h_rec_pratica_mm'] = min(hs) if hs else None
    liv = [riep.get(k) for k in ('A_livello', 'B_livello') if riep.get(k) is not None]
    riep['livello_raccomandazione'] = max(liv) if liv else None
    st = [riep.get(k) for k in ('A_h_stimata_mm', 'B_h_stimata_mm') if riep.get(k) is not None and np.isfinite(riep.get(k))]
    riep['h_stimata_soglia_mm'] = min(st) if st else None
    riep['err_app_max'] = max([riep.get(k) for k in ('A_err_app', 'B_err_app') if riep.get(k) is not None and np.isfinite(riep.get(k))] or [np.nan])


def _json_salva(riep, percorso):
    def conv(o):
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return str(o)
    with open(percorso, 'w') as f:
        json.dump(riep, f, default=conv)


def _nodi(n):
    return (n[0] + 1) * (n[1] + 1) * (n[2] + 1)


def stima_h_predetta(v, opz):
    """Per uno studio che non soddisfa la soglia: dimensione di mesh PREVISTA per raggiungerla, estrapolando con l'ordine osservato
    (limitato a [0.5, 2]; se la convergenza non e' monotona si assume ordine 1). E' una stima, non una verifica."""
    if v.get('idx_ess') is not None or v['det'].empty or not np.isfinite(v.get('h_fine', np.nan)):
        return None
    det, tab = v['det'], v['tab']
    stime = []
    for _, r in det[det.h_min_richiesta_mm.isna()].iterrows():
        e, thr = r['scarto_fine_pc'], opz['soglia']
        if not np.isfinite(e) or e <= thr or e <= 0:
            continue
        t = tab[tab.grandezza == r.grandezza]
        if len(t) and np.isfinite(t.p_osservato.iloc[0]) and t.regime.iloc[0] == 'monotona':
            p = float(np.clip(t.p_osservato.iloc[0], 0.5, 2.0))
        else:
            p = 1.0
        stime.append(v['h_fine'] * (thr / e) ** (1.0 / p))
    return min(stime) if stime else None


def completa_riepilogo(riep, vA, vB, nodi_A_fine, nodi_B_fine, opz):
    """Campi che garantiscono SEMPRE una dimensione di mesh operativa: la raccomandata se la soglia e' raggiunta, altrimenti la piu' fine
    provata con la sua incertezza, piu' la mesh prevista per raggiungere la soglia (se stimabile) e il relativo numero di nodi."""
    hp = [x for x in (stima_h_predetta(vA, opz) if vA['h_ess'] is None else None, stima_h_predetta(vB, opz) if vB['h_ess'] is None else None) if x]
    riep['h_stato'] = 'soglia_raggiunta' if riep.get('h_raccomandata_mm') is not None else 'soglia_non_raggiunta'
    bloc = [t for t, v in (('A', vA), ('B', vB)) if v['h_ess'] is None and not v['tab'].empty]
    non_asint = False
    for tag in bloc:
        v = vA if tag == 'A' else vB
        for g in str(v['bloccante']).split(','):
            r = v['tab'][v['tab'].grandezza == g]
            if len(r) and r.regime.iloc[0] != 'monotona':
                non_asint = True
    if riep['h_stato'] == 'soglia_non_raggiunta' and non_asint:
        riep['h_stato'] = 'non_asintotica'
    riep['h_operativa_mm'] = riep.get('h_raccomandata_mm') if riep.get('h_raccomandata_mm') is not None else riep.get('h_fine_provata_mm')
    ge = [x for x in (riep.get('A_GCI_ess_max'), riep.get('B_GCI_ess_max')) if x is not None and np.isfinite(x)]
    # soglia raggiunta: l'errore alla mesh operativa e' <= soglia per costruzione; altrimenti si riporta il GCI massimo delle essenziali sulla mesh piu' fine
    riep['incertezza_operativa_pc'] = float(opz['soglia']) if riep['h_stato'] == 'soglia_raggiunta' else (max(ge) if ge else None)
    # mesh per l'uso con le TEMPERATURE (calibrazione/validazione): decisa dallo studio B; la A riguarda le grandezze elettriche (ripartizione di potenza)
    riep['h_temperature_mm'] = riep.get('B_h_ess_mm') if riep.get('B_h_ess_mm') is not None else riep.get('B_h_fine_mm')
    riep['h_temperature_stato'] = 'soglia_raggiunta' if riep.get('B_h_ess_mm') is not None else 'soglia_non_raggiunta'
    riep['h_predetta_mm'] = min(hp) if (hp and riep['h_stato'] != 'soglia_raggiunta') else None
    if riep['h_predetta_mm'] and riep.get('h_fine_provata_mm'):
        nf = max(nodi_A_fine or 0, nodi_B_fine or 0)
        riep['nodi_predetti'] = float(nf * (riep['h_fine_provata_mm'] / riep['h_predetta_mm']) ** 3)
    else:
        riep['nodi_predetti'] = None
    return riep


def analizza_caso(caso, cartella, opz):
    """Esegue per un caso gli studi A (elettrico, con raffinamento adattivo), B (accoppiato, a tempo fisso e a tau*), C (passo temporale) e
    S (sensitivita' locale). Salva in cartella/<id>/ e restituisce il riepilogo, che include la GRANDEZZA CHE BLOCCA la soglia in ogni studio."""
    cid, geo, p = caso['id'], caso['geo'], caso['p']
    dirc = os.path.join(cartella, cid); os.makedirs(dirc, exist_ok=True)
    riep = dict(id=cid, ricetta=caso['ricetta'], config=caso['config'], cella=caso['cella'], sigma_m20=p['_sig_m20'], alfa_m=p['_alfa_m'])
    t0 = time.time()
    senza_polpette = len(geo['centri']) == 0
    T0c = p['T_iniziale']
    tol = {g: opz['tol_abs_C'] for g in _QOI_T}
    # ---------------- A: solo elettrico, con raffinamento adattivo ----------------
    GA = griglie_per_geo(geo, opz['h_base_mm'], opz['livelli_A'], opz['max_nodi_A'])
    GA_ext = list(GA)
    dfA, hA = studio_elettrico(geo, GA, p, verbose=False)
    ess_A = [g for g in GRAND_ESS_A if not (senza_polpette and g in ('frac_P_p', 'q_media_polp'))]
    loc_A = [g for g in GRAND_LOC_A if g in dfA.columns]
    vA = valuta_convergenza(dfA, hA, ess_A, loc_A, opz, tol_abs=tol)
    n_adatt = 0
    while opz.get('adattivo_A', True):
        if vA['idx_ess'] is not None and vA['idx_ess'] >= 1:       # soglia raggiunta e confermata da almeno una griglia piu' fine
            break
        nxt = tuple(2 * x for x in GA_ext[-1])
        if _nodi(nxt) > opz['max_nodi_A_adattivo']:
            break
        d1, h1 = studio_elettrico(geo, [nxt], p, verbose=False)
        dfA = pd.concat([dfA, d1], ignore_index=True); hA = np.concatenate([hA, h1]); GA_ext.append(nxt); n_adatt += 1
        vA = valuta_convergenza(dfA, hA, ess_A, loc_A, opz, tol_abs=tol)
    dfA.to_csv(os.path.join(dirc, 'A.csv'), index=False); vA['tab'].to_csv(os.path.join(dirc, 'A_conv.csv'), index=False)
    vA['det'].to_csv(os.path.join(dirc, 'A_dettaglio_soglia.csv'), index=False)
    vAc = None
    if opz.get('studio_corr_volume', True) and not senza_polpette:      # stessa serie con il raggio corretto in modo che l'integrale di phi eguagli il volume delle sfere
        dfAc, hAc = studio_elettrico(geo, GA, p, corr_volume=True, verbose=False)
        vAc = valuta_convergenza(dfAc, hAc, ess_A, loc_A, opz, tol_abs=tol)
        dfAc.to_csv(os.path.join(dirc, 'A_corr_volume.csv'), index=False); vAc['tab'].to_csv(os.path.join(dirc, 'A_corr_volume_conv.csv'), index=False)
    riep.update(A_griglie=len(GA_ext), A_livelli_aggiunti=n_adatt, A_nodi_max=int(dfA.n_nodi.max()), A_h_fine_mm=vA['h_fine'],
                A_h_ess_mm=vA['h_ess'], A_h_loc_mm=vA['h_loc'], A_bloccante=vA['bloccante'], A_scarto_bloccante=vA['scarto_bloccante'],
                A_GCI_ess_max=vA['GCI_ess_max'], A_ident_max=float((dfA.P_meno_VI.abs() / dfA.P).max()), A_sbil_max=float(dfA.sbilancio_I.max()))
    for g in loc_A:
        r = vA['tab'][vA['tab'].grandezza == g]
        riep[f'GCI_A_{g}'] = float(r.GCI_fine_percento.iloc[0]) if not r.empty else np.nan
    # ---------------- B: accoppiato, con pilota sulla griglia piu' grossolana ----------------
    GB = list(GA[:opz['griglie_B']])
    t_fix = opz['t_fisso']
    t_lim = opz.get('T_limite_valutazione')
    t_eval, limitato = t_fix, False
    if t_lim:      # il modello non include l'ebollizione: si valuta al piu' al tempo in cui T_max raggiunge T_limite_valutazione (pilota su mesh grossolana, passo 5 s)
        ctxp = Contesto(geo, GA[min(1, len(GA) - 1)], p, costruisci_filtro=False)
        pp = dict(p); pp['dt'] = 5.0
        _, infp = simula_diag(ctxp, pp, 1e9, T_crit=float(t_lim), durata_max=t_fix + 10, t_min_run=0.0)
        if infp['t_boil'] is not None and infp['t_boil'] < t_fix:
            t_eval = float(max(30.0, np.floor(infp['t_boil'] / 5.0) * 5.0)); limitato = True
        del ctxp
    t_fix = t_eval
    riep.update(t_valutazione=t_eval, t_limitato_ebollizione=limitato)
    usa_tau = bool(opz.get('usa_tau', False))
    dur = opz['durata_max'] if usa_tau else t_fix + 2 * opz['dt']
    stato_pilota = None
    if usa_tau and opz.get('pilota_tau', True) and GB:
        ctx0 = Contesto(geo, GB[0], p, r_filtro=opz['r_filtro_mm'] / 1000.0)
        _, inf0 = simula_diag(ctx0, p, opz['T_target'], durata_max=opz['durata_max'], t_min_run=t_fix)
        stato_pilota = inf0['stato']
        if stato_pilota == 'tau_raggiunto':
            dur = min(opz['durata_max'], 1.25 * inf0['t_tau'] + 2 * opz['dt'])
        elif stato_pilota == 'bollitura_prima_di_tau':
            dur = min(opz['durata_max'], max(t_fix, inf0['t_boil']) + 2 * opz['dt'])
        else:                                                           # tau non raggiungibile: si legge solo a tempo fisso
            dur = t_fix + 2 * opz['dt']
        del ctx0
    righe_tf, righe_tau, h_B, stati = [], [], [], []

    def corri_B(n):
        ctx = Contesto(geo, n, p, r_filtro=opz['r_filtro_mm'] / 1000.0, costruisci_filtro=usa_tau)
        serie, info = simula_diag(ctx, p, opz['T_target'], durata_max=dur, t_min_run=t_fix)
        tf = valori_a_tempo(serie, t_fix)
        tf.update(n_nodi=info['n_nodi'], h_rep_mm=ctx.h_rep * 1000)
        righe_tf.append(tf)
        tau, _ = valori_a_tau(serie, opz['T_target']) if (usa_tau and info['stato'] == 'tau_raggiunto') else ({}, info['stato'])
        tau.update(n_nodi=info['n_nodi'], h_rep_mm=ctx.h_rep * 1000, stato=info['stato'])
        righe_tau.append(tau); stati.append(info['stato']); h_B.append(ctx.h_rep)
        riep['chiusura_max'] = max(riep.get('chiusura_max', 0.0), float(serie.chiusura_energia.abs().max()))
        serie.to_csv(os.path.join(dirc, f'B_serie_{n[0]}_{n[1]}_{n[2]}.csv'), index=False)
        del ctx
    for n in GB:
        corri_B(n)
    # raffinamento adattivo anche del problema accoppiato: si aggiungono livelli finche' la soglia e' raggiunta e confermata (o tetto di nodi)
    GB_cand = [n for n in griglie_per_geo(geo, opz['h_base_mm'], opz['livelli_A'] + 2, opz['max_nodi_B_adattivo']) if n not in GB]
    n_adatt_B = 0
    essB_tmp = [g for g in GRAND_ESS_B if not (senza_polpette and g == 'scarto_pol_pur')]
    GB_run = list(GB)
    while opz.get('adattivo_B', True) and len(righe_tf) >= 3 and GB_cand:
        dfT = pd.DataFrame(righe_tf); hT = np.array(h_B)
        riseT_ = float(dfT['T_max'].values[np.argmin(hT)] - T0c)
        vT = valuta_convergenza(dfT, hT, [g for g in essB_tmp if g in dfT.columns], GRAND_LOC_B, opz, {g: riseT_ for g in _QOI_T}, tol)
        if vT['idx_ess'] is not None and vT['idx_ess'] >= 1:
            break
        nxt = GB_cand.pop(0)
        corri_B(nxt); GB_run.append(nxt); n_adatt_B += 1
    GB = GB_run
    riep['B_livelli_aggiunti'] = n_adatt_B
    dfBf = pd.DataFrame(righe_tf); dfBt = pd.DataFrame(righe_tau); hB = np.array(h_B)
    essB = [g for g in GRAND_ESS_B if g in dfBf.columns and not (senza_polpette and g == 'scarto_pol_pur')]
    rise = float(dfBf['T_max'].values[np.argmin(hB)] - T0c) if len(dfBf) else np.nan
    scB = {g: rise for g in _QOI_T}
    vB = valuta_convergenza(dfBf, hB, essB, GRAND_LOC_B, opz, scB, tol) if len(GB) >= 3 else dict(h_ess=None, h_loc=None, bloccante='', scarto_bloccante=np.nan, GCI_ess_max=np.nan, tab=pd.DataFrame(), det=pd.DataFrame(), h_fine=float(hB.min() * 1000))
    dfBf.to_csv(os.path.join(dirc, 'B_tempo_fisso.csv'), index=False); dfBt.to_csv(os.path.join(dirc, 'B_tau.csv'), index=False)
    vB['tab'].to_csv(os.path.join(dirc, 'B_conv_tempo_fisso.csv'), index=False); vB['det'].to_csv(os.path.join(dirc, 'B_dettaglio_soglia.csv'), index=False)
    riep.update(B_griglie=len(GB), B_h_fine_mm=vB['h_fine'], B_h_ess_mm=vB['h_ess'], B_h_loc_mm=vB['h_loc'], B_bloccante=vB['bloccante'],
                B_scarto_bloccante=vB['scarto_bloccante'], B_GCI_ess_max=vB['GCI_ess_max'], stato_pilota=stato_pilota, durata_eff=dur,
                stato_tau=stati[-1] if (stati and usa_tau) else None, usa_tau=usa_tau, dT_max_tfix=(float(dfBf['T_max'].values[np.argmin(hB)]) - T0c) if (len(dfBf) and 'T_max' in dfBf.columns) else np.nan,
                tau_star=float(dfBt['tau_star'].iloc[0]) if 'tau_star' in dfBt.columns and np.isfinite(dfBt['tau_star'].iloc[0]) else np.nan,
                T_max_a_tau=float(dfBt['T_max'].iloc[0]) if 'T_max' in dfBt.columns and np.isfinite(dfBt['T_max'].iloc[0]) else np.nan,
                T_max_a_tfix=float(dfBf['T_max'].values[np.argmin(hB)]) if len(dfBf) and 'T_max' in dfBf.columns else np.nan, t_fisso=t_fix)
    if usa_tau and len(GB) >= 3 and all(x == 'tau_raggiunto' for x in stati):
        gt = [g for g in GRAND_ESS_B_TAU if g in dfBt.columns and not (senza_polpette and g == 'scarto_pol_pur')]
        riseT = float(dfBt['T_max'].values[np.argmin(hB)] - T0c)
        scBt = {g: riseT for g in _QOI_T}; scBt['E_pm'] = float(dfBt['E_in'].values[np.argmin(hB)]) if 'E_in' in dfBt.columns else riseT
        vBt = valuta_convergenza(dfBt, hB, gt, [], opz, scBt, tol)
        vBt['tab'].to_csv(os.path.join(dirc, 'B_conv_tau.csv'), index=False)
        riep['Btau_h_ess_mm'] = vBt['h_ess']; riep['Btau_bloccante'] = vBt['bloccante']
    # ---------------- C: passo temporale ----------------
    nC = GA[min(opz['griglia_C'], len(GA) - 1)]
    dts = [opz['dt'] * 4, opz['dt'] * 2, opz['dt'], opz['dt'] / 2]
    ctxC = Contesto(geo, nC, p, r_filtro=opz['r_filtro_mm'] / 1000.0, costruisci_filtro=usa_tau)
    righeC = []

    def _un_dt(dt):
        pp = dict(p); pp['dt'] = float(dt)
        serie, info = simula_diag(ctxC, pp, opz['T_target'], durata_max=max(dur, t_fix + 2 * max(dts)), t_min_run=t_fix)
        tf = valori_a_tempo(serie, t_fix); tf.update(dt=dt, stato=info['stato'])
        if usa_tau and info['stato'] == 'tau_raggiunto':
            tau, _ = valori_a_tau(serie, opz['T_target']); tf['tau_star'] = tau['tau_star']
        righeC.append(tf)
    for dt in dts:
        _un_dt(dt)
    n_extra = 0
    while True:                       # se nessun passo soddisfa la soglia (riscaldamento molto rapido) si dimezza ancora il passo, fino a dt_livelli_extra volte
        dfC = pd.DataFrame(righeC)
        gC = [g for g in ['T_media_polp', 'T_media_pur', 'scarto_pol_pur', 'T_P1', 'T_P2', 'T_P3', 'E_in', 'tau_star'] if g in dfC.columns and not (senza_polpette and g == 'scarto_pol_pur')]
        gC = [g for g in gC if np.all(np.isfinite(dfC[g].values))]
        riseC = float(dfC['T_max'].values[np.argmin(dfC['dt'].values)] - T0c)
        vC = valuta_convergenza(dfC, dfC['dt'].values / 1000.0, gC, ['T_max'], opz, {g: riseC for g in _QOI_T}, tol)   # dt in 'm' fittizi: si usa solo l'ordinamento
        if vC['h_ess'] is not None or n_extra >= opz.get('dt_livelli_extra', 3):
            break
        _un_dt(float(dfC['dt'].min()) / 2.0); n_extra += 1
    riep['C_dt_livelli_extra'] = n_extra
    dfC.to_csv(os.path.join(dirc, 'C.csv'), index=False); vC['tab'].to_csv(os.path.join(dirc, 'C_conv.csv'), index=False)
    dtm = None if vC['h_ess'] is None else float(vC['h_ess'])                                              # h_ess e' gia' in "mm" = dt*1000/1000
    riep['C_dt_massimo_provato'] = bool(dtm is not None and dtm >= float(dfC['dt'].max()))
    riep['C_dt_rifiutato'] = next((float(x) for x in sorted(dfC['dt'].values) if dtm is not None and x > dtm), None)
    riep['C_dt_provati'] = ', '.join(f'{x:g}' for x in sorted(dfC['dt'].values, reverse=True))
    riep.update(C_dt_min=dtm, C_dt_fine=float(dfC['dt'].min()), C_nodi=int(ctxC.basis.N), C_bloccante=vC['bloccante'], C_scarto_bloccante=vC['scarto_bloccante'])
    del ctxC
    # ---------------- mesh e passo raccomandati ----------------
    hs = [x for x in (riep['A_h_ess_mm'], riep['B_h_ess_mm']) if x is not None]
    riep['h_raccomandata_mm'] = min(hs) if (len(hs) == (2 if len(GB) >= 3 else 1) and hs) else None
    hl = [x for x in (riep['A_h_loc_mm'], riep['B_h_loc_mm']) if x is not None]
    riep['h_con_locali_mm'] = min(hl) if (len(hl) == (2 if len(GB) >= 3 else 1) and hl) else None
    riep['h_fine_provata_mm'] = min(x for x in (riep['A_h_fine_mm'], riep['B_h_fine_mm']) if x is not None and np.isfinite(x))
    _aggiungi_raccomandazione(riep, vA, vB if len(GB) >= 3 else None, vC, vAc)
    completa_riepilogo(riep, vA, vB, int(dfA.n_nodi.max()), int(dfBf.n_nodi.max()) if len(dfBf) else 0, opz)
    # ---------------- S: sensitivita' locale su mesh economica ----------------
    if opz['esegui_S']:
        hr = riep['h_raccomandata_mm'] or float(np.sort(hA)[len(hA) // 2] * 1000)
        nS = GA[0]
        for n in GA:
            h_rep_n = (geo['Lx'] * geo['Ly'] * geo['Lz'] / (n[0] * n[1] * n[2])) ** (1.0 / 3.0) * 1000
            if h_rep_n <= hr + 1e-9 and _nodi(n) <= opz['max_nodi_S']:
                nS = n
        tabS, X, selez = sensibilita_locale(geo, p, nS, opz['dt_S'], opz['durata_misure'], opz['delta'], opz['passo_misura'], opz['sigma_misura'], opz['se_max'])
        tabS.to_csv(os.path.join(dirc, 'S.csv'), index=False)
        riep.update(S_mesh=str(nS), S_sel_tutte=','.join(selez.get('tutte', [])), S_sel_purea=','.join(selez.get('purea', [])), S_sel_polpetta=','.join(selez.get('polpetta', [])))
    modo_netto = opz.get('riferimento_netto', 'elettrico')
    if modo_netto in ('elettrico', 'completo') and not senza_polpette:
        try:
            riferimento_netto_caso(caso, cartella, opz, accoppiato=(modo_netto == 'completo'))
        except Exception as e:
            riep['netto_errore'] = str(e)[:120]
    riep['tempo_s'] = time.time() - t0
    _json_salva(riep, os.path.join(dirc, 'riepilogo.json'))
    return riep


def ricalcola_caso(d, cid, o, T0=20.0):
    """Ricalcola dai file salvati in d (A.csv, B_tempo_fisso.csv, C.csv) mesh/passo minimi, grandezze bloccanti, incertezze e raccomandazioni di un caso, e aggiorna d/riepilogo.json."""
    tol = {g: o['tol_abs_C'] for g in _QOI_T}
    f = os.path.join(d, 'riepilogo.json')
    riep = json.load(open(f))
    senza = cid.endswith('__RP')
    fA, fB, fC = (os.path.join(d, x) for x in ('A.csv', 'B_tempo_fisso.csv', 'C.csv'))
    if os.path.exists(fA):
        A = pd.read_csv(fA); hA = A['h_rep_mm'].values / 1000.0
        ess = [g for g in GRAND_ESS_A if not (senza and g in ('frac_P_p', 'q_media_polp'))]
        vA = valuta_convergenza(A, hA, ess, [g for g in GRAND_LOC_A if g in A.columns], o, tol_abs=tol)
        vA['tab'].to_csv(os.path.join(d, 'A_conv.csv'), index=False); vA['det'].to_csv(os.path.join(d, 'A_dettaglio_soglia.csv'), index=False)
        riep.update(A_h_fine_mm=vA['h_fine'], A_h_ess_mm=vA['h_ess'], A_h_loc_mm=vA['h_loc'], A_bloccante=vA['bloccante'],
                    A_scarto_bloccante=vA['scarto_bloccante'], A_GCI_ess_max=vA['GCI_ess_max'])
    if os.path.exists(fB):
        B = pd.read_csv(fB); hB = B['h_rep_mm'].values / 1000.0
        if len(B) >= 3:
            rise = float(B['T_max'].values[np.argmin(hB)] - T0)
            essB = [g for g in GRAND_ESS_B if g in B.columns and not (senza and g == 'scarto_pol_pur')]
            vB = valuta_convergenza(B, hB, essB, GRAND_LOC_B, o, {g: rise for g in _QOI_T}, tol)
            vB['tab'].to_csv(os.path.join(d, 'B_conv_tempo_fisso.csv'), index=False); vB['det'].to_csv(os.path.join(d, 'B_dettaglio_soglia.csv'), index=False)
            riep.update(dT_max_tfix=rise, T_max_a_tfix=rise + T0, B_h_fine_mm=vB['h_fine'], B_h_ess_mm=vB['h_ess'], B_h_loc_mm=vB['h_loc'], B_bloccante=vB['bloccante'],
                        B_scarto_bloccante=vB['scarto_bloccante'], B_GCI_ess_max=vB['GCI_ess_max'])
    if os.path.exists(fC):
        C = pd.read_csv(fC)
        gC = [g for g in ['T_media_polp', 'T_media_pur', 'scarto_pol_pur', 'T_P1', 'T_P2', 'T_P3', 'E_in', 'tau_star'] if g in C.columns and not (senza and g == 'scarto_pol_pur')]
        gC = [g for g in gC if np.all(np.isfinite(C[g].values))]
        riseC = float(C['T_max'].values[np.argmin(C['dt'].values)] - T0)
        vC = valuta_convergenza(C, C['dt'].values / 1000.0, gC, ['T_max'], o, {g: riseC for g in _QOI_T}, tol)
        vC['tab'].to_csv(os.path.join(d, 'C_conv.csv'), index=False)
        riep['C_dt_massimo_provato'] = bool(vC['h_ess'] is not None and float(vC['h_ess']) >= float(C['dt'].max()))
        _d = None if vC['h_ess'] is None else float(vC['h_ess'])
        riep['C_dt_rifiutato'] = next((float(x) for x in sorted(C['dt'].values) if _d is not None and x > _d), None)
        riep['C_dt_provati'] = ', '.join(f'{x:g}' for x in sorted(C['dt'].values, reverse=True))
        riep.update(C_dt_min=None if vC['h_ess'] is None else float(vC['h_ess']), C_dt_fine=float(C['dt'].min()), C_bloccante=vC['bloccante'],
                    C_scarto_bloccante=vC['scarto_bloccante'])
    hs = [x for x in (riep.get('A_h_ess_mm'), riep.get('B_h_ess_mm')) if x is not None]
    riep['h_raccomandata_stretta_mm'] = min(hs) if len(hs) == 2 else None
    riep['h_raccomandata_mm'] = riep['h_raccomandata_stretta_mm']
    hl = [x for x in (riep.get('A_h_loc_mm'), riep.get('B_h_loc_mm')) if x is not None]
    riep['h_con_locali_mm'] = min(hl) if len(hl) == 2 else None
    fin = [x for x in (riep.get('A_h_fine_mm'), riep.get('B_h_fine_mm')) if x is not None]
    riep['h_fine_provata_mm'] = min(fin) if fin else None
    if os.path.exists(fA) and os.path.exists(fB) and len(B) >= 3:
        completa_riepilogo(riep, vA, vB, int(A.n_nodi.max()), int(B.n_nodi.max()), o)
    _aggiungi_raccomandazione(riep, vA if os.path.exists(fA) else None, vB if (os.path.exists(fB) and len(B) >= 3) else None, vC if os.path.exists(fC) else None, None)
    _json_salva(riep, f)
    return riep


def ricalcola_risultati(cartella, opz, T0=20.0, verbose=True):
    """Ricalcola mesh e passo minimi, grandezze bloccanti e incertezze dai file GIA' SALVATI (A.csv, B_tempo_fisso.csv, C.csv, ...),
    con i criteri correnti (soglia, soglia assoluta). Non riesegue nessuna simulazione. Aggiorna i riepilogo.json."""
    o = dict(OPZ_DEFAULT); o.update(opz or {})
    base = os.path.join(cartella, 'casi'); n = 0
    tol = {g: o['tol_abs_C'] for g in _QOI_T}
    for cid in sorted(os.listdir(base)):
        d = os.path.join(base, cid); f = os.path.join(d, 'riepilogo.json')
        if not os.path.exists(f):
            continue
        riep = ricalcola_caso(d, cid, o, T0); n += 1
        if verbose:
            print(f"{cid}: h operativa = {riep.get('h_operativa_mm')} mm ({riep.get('h_stato')}) | incertezza = {riep.get('incertezza_operativa_pc')} % | h prevista = {riep.get('h_predetta_mm')} | bloccanti A: {riep.get('A_bloccante')} | B: {riep.get('B_bloccante')} | dt min = {riep.get('C_dt_min')}")
    return n


OPZ_DEFAULT = dict(h_base_mm=10.0, livelli_A=4, max_nodi_A=150000, adattivo_A=True, max_nodi_A_adattivo=900000, adattivo_B=True, max_nodi_B_adattivo=130000, griglie_B=3, griglia_C=1, max_nodi_app=35000, studio_corr_volume=True, riferimento_netto='elettrico', h_prod_mm=2.5, T_limite_valutazione=90.0, dt_livelli_extra=3,
                   soglia=2.0, tol_abs_C=0.2, usa_tau=False, T_target=T_COTTURA_C, t_fisso=270.0, durata_max=1500.0, r_filtro_mm=3.0, dt=10.0, pilota_tau=True,
                   esegui_S=True, max_nodi_S=3000, dt_S=10.0, delta=0.10, passo_misura=5.0, durata_misure=270.0, sigma_misura=0.5, se_max=0.20)


# ---------------------------------------------------------------------------------------------------------
# Google Drive: checkpoint e sincronizzazione dei risultati
# ---------------------------------------------------------------------------------------------------------
import datetime
import shutil


def monta_drive(punto='/content/drive'):
    """Monta Google Drive in Colab (richiede l'autorizzazione dell'utente). Restituisce True se disponibile."""
    try:
        from google.colab import drive
        drive.mount(punto)
        return os.path.isdir(os.path.join(punto, 'MyDrive'))
    except Exception as e:
        print("Google Drive non disponibile:", e)
        return False


def _copia_dir(src, dst):
    """Copia una cartella escludendo i file di mesh (*.msh, pesanti e rigenerabili) e copiando per ULTIMO riepilogo.json: se la copia viene interrotta, il caso non risulta completo."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.msh', 'riepilogo.json'))
    r = os.path.join(src, 'riepilogo.json')
    if os.path.exists(r):
        shutil.copy2(r, os.path.join(dst, 'riepilogo.json'))


def casi_completati(cartella):
    base = os.path.join(cartella, 'casi')
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.exists(os.path.join(base, d, 'riepilogo.json')) and os.path.exists(os.path.join(base, d, 'A.csv')))


def scrivi_checkpoint(cartella, opz=None, casi_previsti=None, extra=None):
    """Scrive cartella/checkpoint.json: casi completati, casi previsti, opzioni e data. Serve a riprendere e a sapere a che punto si e'."""
    ck = dict(revisione='rev. 4', aggiornato=datetime.datetime.now().isoformat(timespec='seconds'), completati=casi_completati(cartella),
              previsti=list(casi_previsti or []), opzioni={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in (opz or {}).items()})
    if extra:
        ck.update(extra)
    with open(os.path.join(cartella, 'checkpoint.json'), 'w') as f:
        json.dump(ck, f, indent=1)
    return ck


def ripristina_da_drive(cartella_locale, cartella_drive, verbose=True):
    """Copia dal Drive nella cartella locale i casi completati che mancano in locale (riprendere dopo il riavvio di Colab)."""
    base_d, base_l = os.path.join(cartella_drive, 'casi'), os.path.join(cartella_locale, 'casi')
    n = 0
    if os.path.isdir(base_d):
        da_fare = [c for c in sorted(os.listdir(base_d)) if os.path.exists(os.path.join(base_d, c, 'riepilogo.json')) and not (os.path.exists(os.path.join(base_l, c, 'riepilogo.json')) and os.path.exists(os.path.join(base_l, c, 'A.csv')))]
        t0 = time.time()
        for k, cid in enumerate(da_fare, 1):
            _copia_dir(os.path.join(base_d, cid), os.path.join(base_l, cid)); n += 1
            if verbose:
                print(f"  ripristino dal Drive: {k}/{len(da_fare)} ({cid}, {time.time() - t0:.0f} s)", flush=True)
    for nome in ('errori.csv', 'checkpoint.json'):
        pd_, pl_ = os.path.join(cartella_drive, nome), os.path.join(cartella_locale, nome)
        if os.path.exists(pd_) and not os.path.exists(pl_):
            os.makedirs(cartella_locale, exist_ok=True); shutil.copy2(pd_, pl_)
    if verbose:
        print(f"Ripristinati dal Drive {n} casi completati")
    return n


def sincronizza_su_drive(cartella_locale, cartella_drive, casi=None, includi_output=False, verbose=False):
    """Copia sul Drive i casi indicati (o tutti), il checkpoint, gli errori e, se includi_output, anche report, Excel, figure e archivi.
    Gli errori di copia non interrompono il calcolo."""
    try:
        os.makedirs(cartella_drive, exist_ok=True)
        for cid in (casi if casi is not None else casi_completati(cartella_locale)):
            src = os.path.join(cartella_locale, 'casi', cid)
            if os.path.isdir(src):
                _copia_dir(src, os.path.join(cartella_drive, 'casi', cid))
        for nome in ('checkpoint.json', 'errori.csv'):
            if os.path.exists(os.path.join(cartella_locale, nome)):
                shutil.copy2(os.path.join(cartella_locale, nome), os.path.join(cartella_drive, nome))
        if includi_output:
            for nome in sorted(os.listdir(cartella_locale)):
                p_ = os.path.join(cartella_locale, nome)
                if nome == 'casi' or nome in ('checkpoint.json', 'errori.csv'):
                    continue
                if os.path.isdir(p_):
                    _copia_dir(p_, os.path.join(cartella_drive, nome))
                else:
                    shutil.copy2(p_, os.path.join(cartella_drive, nome))
        if verbose:
            print("Sincronizzato su Drive:", cartella_drive)
        return True
    except Exception as e:
        print("ATTENZIONE: sincronizzazione su Drive non riuscita:", e)
        return False


def messaggio_dt(riep):
    """Testo non ambiguo sul passo temporale: dice sempre quali passi sono stati provati e come stanno rispetto alla soglia."""
    dtm, prov = riep.get('C_dt_min'), riep.get('C_dt_provati', '')
    if dtm is None:
        return f"nessuno dei passi provati ({prov} s) soddisfa la soglia"
    if riep.get('C_dt_massimo_provato'):
        return f"dt = {dtm:g} s soddisfa la soglia ed e' il PIU' GRANDE provato (provati {prov} s): il limite reale non e' stato trovato, vale >= {dtm:g} s"
    return f"dt = {dtm:g} s soddisfa la soglia; il passo successivo {riep.get('C_dt_rifiutato'):g} s no (provati {prov} s): il limite reale e' tra {dtm:g} e {riep.get('C_dt_rifiutato'):g} s"


def esegui_tutti(ricette_sel, config_sel, cartella='risultati_sensitivita', opz=None, campo_V_cm=4.0, T0=20.0, T_amb=21.0, riprendi=True,
                 verbose=True, cartella_drive=None, analizza=None):
    """Itera su tutte le combinazioni (ricetta, configurazione). Salva ogni caso su disco locale e, se cartella_drive e' indicata, lo copia
    subito sul Drive insieme al checkpoint. Se riprendi=True salta i casi gia' completati (anche quelli ripristinati dal Drive).
    Un errore in un caso viene registrato in errori.csv e non ferma gli altri."""
    o = dict(OPZ_DEFAULT); o.update(opz or {})
    ric = registro_ricette()
    os.makedirs(os.path.join(cartella, 'casi'), exist_ok=True)
    if cartella_drive:
        ripristina_da_drive(cartella, cartella_drive, verbose)
    casi = [(r, c) for r in ricette_sel for c in config_sel]
    previsti = [f'{r}__{c}' for r, c in casi]
    scrivi_checkpoint(cartella, o, previsti)
    log = []
    for k, (r, c) in enumerate(casi, 1):
        cid = f'{r}__{c}'
        f_ok = os.path.join(cartella, 'casi', cid, 'riepilogo.json')
        if riprendi and os.path.exists(f_ok) and os.path.exists(os.path.join(cartella, 'casi', cid, 'A.csv')):
            if verbose:
                print(f"[{k}/{len(casi)}] {cid}: gia' completato, saltato")
            continue
        t0 = time.time()
        try:
            caso = costruisci_caso(r, c, ric, campo_V_cm, T0, T_amb, o['dt'], o['durata_max'])
            riep = (analizza or analizza_caso)(caso, os.path.join(cartella, 'casi'), o)
            if verbose:
                print(f"[{k}/{len(casi)}] {cid}: ok in {time.time() - t0:.0f} s | h operativa = {riep.get('h_operativa_mm')} mm ({riep.get('h_stato')}) | "
                      f"incertezza = {riep.get('incertezza_operativa_pc')} % | {messaggio_dt(riep)}")
            scrivi_checkpoint(cartella, o, previsti)
            if cartella_drive:
                sincronizza_su_drive(cartella, cartella_drive, casi=[cid])
        except Exception as e:
            log.append(dict(id=cid, errore=str(e), traceback=traceback.format_exc()))
            if verbose:
                print(f"[{k}/{len(casi)}] {cid}: ERRORE: {e}")
            pd.DataFrame(log).to_csv(os.path.join(cartella, 'errori.csv'), index=False)
            if cartella_drive:
                sincronizza_su_drive(cartella, cartella_drive, casi=[])
    if log:
        pd.DataFrame(log).to_csv(os.path.join(cartella, 'errori.csv'), index=False)
    scrivi_checkpoint(cartella, o, previsti, extra=dict(terminato=datetime.datetime.now().isoformat(timespec='seconds'), errori=len(log)))
    if cartella_drive:
        sincronizza_su_drive(cartella, cartella_drive, casi=[])
    return log


def confronto_conforme_caso(caso, cartella, opz, lc_list=((8e-3, 4e-3), (5e-3, 2.5e-3), (3.5e-3, 1.75e-3))):
    """Riferimento a interfaccia netta (mesh conforme alle sfere, gmsh) per il solo problema elettrico; confronta le grandezze con quelle della
    griglia strutturata piu' fine del caso. Salva E_conforme.csv e E_confronto.csv in cartella/<id>/."""
    geo, p = caso['geo'], caso['p']
    dirc = os.path.join(cartella, caso['id'])
    if len(geo['centri']) == 0:
        return None
    A = pd.read_csv(os.path.join(dirc, 'A.csv')).sort_values('h_rep_mm')
    fine = A.iloc[0]
    righe = []
    for lc, lci in lc_list:
        f = mesh_conforme_gmsh(geo, lc, lci, os.path.join(dirc, 'conforme.msh'))
        r = elettrico_conforme(geo, p, f)
        r.update(lc_mm=lc * 1e3, lc_interfaccia_mm=lci * 1e3)
        righe.append(r)
    df = pd.DataFrame(righe)
    df.to_csv(os.path.join(dirc, 'E_conforme.csv'), index=False)
    conf = []
    for g in ['P', 'frac_P_p', 'q_media_polp', 'q_media_pur', 'q_strati_z', 'q_asse', 'q_p99']:
        if g in df.columns and g in fine.index and np.isfinite(df[g].iloc[-1]) and df[g].iloc[-1] != 0:
            conf.append(dict(grandezza=g, strutturata_fine=float(fine[g]), conforme_fine=float(df[g].iloc[-1]),
                             differenza_percento=float(100 * (fine[g] - df[g].iloc[-1]) / df[g].iloc[-1]),
                             nodi_strutturata=int(fine['n_nodi']), nodi_conforme=int(df['n_nodi'].iloc[-1])))
    c = pd.DataFrame(conf)
    c.to_csv(os.path.join(dirc, 'E_confronto.csv'), index=False)
    return c


def esegui_conformi(cartella, opz=None, campo_V_cm=4.0, T0=20.0, T_amb=21.0, solo_senza_soglia=True, verbose=True):
    """Esegue il riferimento conforme per i casi in cui la soglia non e' stata raggiunta (o per tutti). Richiede gmsh e meshio."""
    o = dict(OPZ_DEFAULT); o.update(opz or {})
    ric = registro_ricette()
    base = os.path.join(cartella, 'casi'); n = 0
    for cid in sorted(os.listdir(base)):
        f = os.path.join(base, cid, 'riepilogo.json')
        if not os.path.exists(f):
            continue
        riep = json.load(open(f))
        if solo_senza_soglia and riep.get('h_raccomandata_mm') is not None:
            continue
        if os.path.exists(os.path.join(base, cid, 'E_confronto.csv')):
            continue
        try:
            caso = costruisci_caso(riep['ricetta'], riep['config'], ric, campo_V_cm, T0, T_amb, o['dt'], o['durata_max'])
            c = confronto_conforme_caso(caso, base, o)
            n += 1
            if verbose and c is not None:
                print(f"{cid}: " + ', '.join(f"{r.grandezza} {r.differenza_percento:+.2f}%" for r in c.itertuples()))
        except Exception as e:
            print(f"{cid}: errore nel riferimento conforme: {e}")
    return n


def frequenza_bloccanti(R):
    """Quante volte ciascuna grandezza blocca la soglia (casi senza soglia raggiunta) o e' la piu' esigente (casi con soglia raggiunta)."""
    righe = []
    for studio, col_b, col_h in (('A elettrico', 'A_bloccante', 'A_h_ess_mm'), ('B accoppiato', 'B_bloccante', 'B_h_ess_mm'), ('C passo temporale', 'C_bloccante', 'C_dt_min')):
        if col_b not in R.columns:
            continue
        for _, r in R.iterrows():
            b = r.get(col_b)
            if not isinstance(b, str) or not b:
                continue
            for g in b.split(','):
                righe.append(dict(studio=studio, grandezza=g, esito='soglia non raggiunta' if pd.isna(r.get(col_h)) else 'piu\' esigente'))
    if not righe:
        return pd.DataFrame()
    return pd.DataFrame(righe).groupby(['studio', 'esito', 'grandezza']).size().rename('casi').reset_index().sort_values(['studio', 'esito', 'casi'], ascending=[True, True, False])


# =========================================================================================================
# 11. AGGREGAZIONE, GRAFICI E REPORT FINALE
# =========================================================================================================
def carica_risultati(cartella):
    """Legge tutti i casi salvati e restituisce un dizionario di DataFrame (con colonne id, ricetta, config)."""
    base = os.path.join(cartella, 'casi')
    riep, A, Bf, Bt, C, S, Ac, Bc, Ec = [], [], [], [], [], [], [], [], []
    for cid in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        d = os.path.join(base, cid)
        f = os.path.join(d, 'riepilogo.json')
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        for k, v in list(r.items()):
            if v in ('None', 'nan'):
                r[k] = None
        riep.append(r)
        ric, conf = cid.split('__')
        def leggi(nome):
            p_ = os.path.join(d, nome)
            if not os.path.exists(p_):
                return None
            x = pd.read_csv(p_)
            x.insert(0, 'config', conf); x.insert(0, 'ricetta', ric); x.insert(0, 'id', cid)
            return x
        for lista, nome in ((A, 'A.csv'), (Bf, 'B_tempo_fisso.csv'), (Bt, 'B_tau.csv'), (C, 'C.csv'), (S, 'S.csv'), (Ac, 'A_conv.csv'), (Bc, 'B_conv_tempo_fisso.csv'), (Ec, 'E_confronto.csv')):
            x = leggi(nome)
            if x is not None:
                lista.append(x)
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()
    R = pd.DataFrame(riep)
    for c in R.columns:
        if not (c in ('id', 'ricetta', 'config', 'cella', 'stato_tau', 'stato_pilota', 'h_stato', 'h_temperature_stato', 'S_mesh', 'S_sel_tutte', 'S_sel_purea', 'S_sel_polpetta') or c.endswith('bloccante')):
            R[c] = pd.to_numeric(R[c], errors='coerce')
    err = os.path.join(cartella, 'errori.csv')
    return dict(riepilogo=R, A=cat(A), B_tempo_fisso=cat(Bf), B_tau=cat(Bt), C=cat(C), S=cat(S), A_conv=cat(Ac), B_conv=cat(Bc), E=cat(Ec),
                errori=pd.read_csv(err) if os.path.exists(err) else pd.DataFrame())


def raccomandazione_globale(R):
    """Per ogni cella: dimensione di mesh e passo temporale piu' restrittivi fra tutti i casi (le grandezze essenziali), con i casi limitanti."""
    righe = []
    for cella, g in R.groupby('cella'):
        h = g.dropna(subset=['h_raccomandata_mm'])
        nd = g[g.h_raccomandata_mm.isna()]
        lim = h.sort_values('h_raccomandata_mm').head(3)
        righe.append(dict(cella=cella, casi=len(g), casi_con_soglia_raggiunta=len(h), casi_senza_soglia=len(nd),
                          h_raccomandata_mm=float(h.h_raccomandata_mm.min()) if len(h) else np.nan,
                          h_mediana_mm=float(h.h_raccomandata_mm.median()) if len(h) else np.nan,
                          dt_raccomandato_s=float(g.C_dt_min.min()) if g.C_dt_min.notna().any() else np.nan,
                          h_operativa_min_mm=float(g.h_operativa_mm.min()) if 'h_operativa_mm' in g.columns and g.h_operativa_mm.notna().any() else np.nan,
                          incertezza_max_pc=float(g.incertezza_operativa_pc.max()) if 'incertezza_operativa_pc' in g.columns and g.incertezza_operativa_pc.notna().any() else np.nan,
                          casi_non_asintotici=int((g.h_stato == 'non_asintotica').sum()) if 'h_stato' in g.columns else 0,
                          casi_limitanti='; '.join(lim.id.tolist()),
                          casi_senza_soglia_elenco='; '.join(nd.id.tolist()[:12])))
    return pd.DataFrame(righe)


def _heat(ax, M, righe, colonne, titolo, fmt='{:.2g}', cmap='YlOrRd', vmin=None, vmax=None, marca=None, fs=10):
    import matplotlib.pyplot as plt
    A = np.array(M, float)
    im = ax.imshow(np.where(np.isfinite(A), A, np.nan), cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(colonne))); ax.set_xticklabels(colonne, rotation=45, ha='right', fontsize=fs)
    ax.set_yticks(range(len(righe))); ax.set_yticklabels(righe, fontsize=fs)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isfinite(A[i, j]):
                t = fmt.format(A[i, j]) + ('*' if (marca is not None and marca[i][j]) else '')
                ax.text(j, i, t, ha='center', va='center', fontsize=fs, color='black', fontweight='bold' if (marca is not None and marca[i][j]) else 'normal')
    ax.set_title(titolo, fontsize=fs + 2)
    return im


def grafici_riepilogo(dati, cartella, soglia=2.0):
    """Produce i grafici di sintesi e li salva in cartella/figure/. Restituisce il dizionario nome -> percorso."""
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11})
    R = dati['riepilogo']
    fig_dir = os.path.join(cartella, 'figure'); os.makedirs(fig_dir, exist_ok=True)
    out = {}
    ricette = sorted(R.ricetta.unique()); configs = [c for c in CONFIGURAZIONI if c in R.config.unique()]

    # 1-3. Una coppia di figure per ricetta (leggibili a larghezza di pagina): convergenza (A e B impilati) e sensitivita'
    Aw = dati['A_conv'].pivot_table(index=['ricetta', 'config'], columns='grandezza', values='GCI_fine_percento').reset_index() if len(dati['A_conv']) else pd.DataFrame()
    Bw = dati['B_conv'].pivot_table(index=['ricetta', 'config'], columns='grandezza', values='GCI_fine_percento').reset_index() if len(dati['B_conv']) else pd.DataFrame()
    S = dati['S']
    if len(S):
        St = S[S.sonde == 'tutte'].pivot_table(index=['id', 'ricetta', 'config'], columns='parametro', values='effetto_10pc_C').reset_index()
        sel = S[S.sonde == 'tutte'].groupby('id').apply(lambda g: ','.join(g[g.selezionato].parametro)).rename('sel').reset_index()
        St = St.merge(sel, on='id')
    else:
        St = pd.DataFrame()
    colsA = ['P', 'frac_P_p', 'q_media_polp', 'q_media_pur', 'q_strati_z', 'q_asse', 'q_p99']
    colsB = ['T_max', 'scarto_pol_pur', 'T_P1', 'T_P2', 'T_P3', 'E_in', 'q_asse', 'q_p99']

    def matrice(df, ric, cols, marca_col=None):
        M = np.full((len(configs), len(cols)), np.nan); Mk = [[False] * len(cols) for _ in configs]
        d = df[df.ricetta == ric] if len(df) else df
        for i, cf in enumerate(configs):
            r = d[d.config == cf] if len(d) else d
            if len(r) == 0:
                continue
            for j, vc in enumerate(cols):
                if vc in r.columns:
                    M[i, j] = r[vc].iloc[0]
                    if marca_col is not None:
                        Mk[i][j] = vc in str(r[marca_col].iloc[0]).split(',')
        return M, Mk

    for ric in ricette:
        fig, axs = plt.subplots(2, 1, figsize=(10.5, 2 * (0.62 * len(configs) + 2.3)))
        for ax, df, cols, tit in ((axs[0], Aw, colsA, 'GCI [%] sulla griglia fine - grandezze elettriche (studio A)'),
                                  (axs[1], Bw, colsB, 'GCI [%] sulla griglia fine - grandezze a tempo fisso (studio B)')):
            M, _ = matrice(df, ric, cols)
            _heat(ax, M, configs, cols, tit, '{:.2g}', 'YlOrRd', 0, 10, fs=12)
        fig.suptitle(ric, fontsize=15); fig.tight_layout(rect=(0, 0, 1, 0.97))
        p_ = os.path.join(fig_dir, f'conv__{ric}.png'); fig.savefig(p_, dpi=130, bbox_inches='tight'); plt.close(fig); out[f'conv:{ric}'] = p_
        if len(St):
            cs = [c for c in PARAMETRI_CAL if c in St.columns]
            M, Mk = matrice(St, ric, cs, 'sel')
            fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(configs) + 2.8))
            _heat(ax, M, configs, [NOMI_PARAM[c].replace(' (fattore)', '') for c in cs], f"{ric}: variazione di T alle sonde [C] per +10% del parametro (* = consigliato)",
                  '{:.2g}', 'Blues', 0, None, marca=Mk, fs=12)
            fig.tight_layout(); p_ = os.path.join(fig_dir, f'sens__{ric}.png'); fig.savefig(p_, dpi=130, bbox_inches='tight'); plt.close(fig); out[f'sens:{ric}'] = p_
    # 4. fattibilita' del criterio di cottura (solo se usato) oppure incremento di temperatura al tempo fisso
    n = len(ricette)
    usa_tau = ('stato_tau' in R.columns) and R.stato_tau.notna().any()
    if usa_tau:
        M = np.full((n, len(configs)), np.nan); T = [[''] * len(configs) for _ in range(n)]
        for i, ric in enumerate(ricette):
            for j, cf in enumerate(configs):
                r = R[(R.ricetta == ric) & (R.config == cf)]
                if r.empty:
                    continue
                st = r.stato_tau.iloc[0]
                if st == 'tau_raggiunto':
                    M[i, j] = 1; T[i][j] = f"{r.tau_star.iloc[0]:.0f} s\nTmax {r.T_max_a_tau.iloc[0]:.0f}C"
                elif st == 'bollitura_prima_di_tau':
                    M[i, j] = 0; T[i][j] = 'bolle\nprima'
                else:
                    M[i, j] = 0.5; T[i][j] = 'non\nraggiunto'
        fig, ax = plt.subplots(figsize=(1.9 * len(configs) + 3, 0.62 * n + 2.2))
        ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(configs))); ax.set_xticklabels(configs, fontsize=12); ax.set_yticks(range(n)); ax.set_yticklabels(ricette, fontsize=12)
        for i in range(n):
            for j in range(len(configs)):
                ax.text(j, i, T[i][j], ha='center', va='center', fontsize=10)
        ax.set_title(f"Criterio di cottura (punto freddo della polpetta a {OPZ_DEFAULT['T_target']:g} C): tempo tau* e Tmax, oppure esito", fontsize=13)
        nome4 = 'G4_fattibilita.png'
    else:
        M = np.full((n, len(configs)), np.nan)
        for i, ric in enumerate(ricette):
            for j, cf in enumerate(configs):
                r = R[(R.ricetta == ric) & (R.config == cf)]
                if not r.empty and 'dT_max_tfix' in r.columns and pd.notna(r['dT_max_tfix'].iloc[0]):
                    M[i, j] = float(r['dT_max_tfix'].iloc[0])
        fig, ax = plt.subplots(figsize=(1.9 * len(configs) + 3, 0.62 * n + 2.2))
        _heat(ax, M, ricette, configs, 'Incremento massimo di temperatura al tempo fisso [C]', '{:.1f}', 'YlOrRd', fs=12)
        nome4 = 'G4_incremento_T.png'
    fig.tight_layout(); p_ = os.path.join(fig_dir, nome4); fig.savefig(p_, dpi=130, bbox_inches='tight'); plt.close(fig); out[nome4] = p_
    # 5. mesh e passo raccomandati
    for col, nome, tit, fmt in (('h_raccomandata_mm', 'G5_mesh.png', 'Dimensione di mesh raccomandata h [mm] (grandezze essenziali; vuoto = soglia non raggiunta)', '{:.1f}'),
                                ('C_dt_min', 'G6_dt.png', 'Passo temporale massimo accettabile [s] (vuoto = soglia non raggiunta)', '{:.0f}')):
        M = np.full((n, len(configs)), np.nan)
        for i, ric in enumerate(ricette):
            for j, cf in enumerate(configs):
                r = R[(R.ricetta == ric) & (R.config == cf)]
                if not r.empty and pd.notna(r[col].iloc[0]):
                    M[i, j] = float(r[col].iloc[0])
        fig, ax = plt.subplots(figsize=(1.9 * len(configs) + 3, 0.62 * n + 2.2))
        _heat(ax, M, ricette, configs, tit, fmt, 'viridis_r', fs=11)
        fig.tight_layout(); p_ = os.path.join(fig_dir, nome); fig.savefig(p_, dpi=130, bbox_inches='tight'); plt.close(fig); out[nome] = p_
    return out


def tabella_parametri_calibrazione(dati):
    """Per ogni ricetta, quante volte (sulle configurazioni) ogni parametro e' consigliato per la calibrazione, con sonde 'tutte'."""
    S = dati['S']
    if S.empty:
        return pd.DataFrame()
    s = S[S.sonde == 'tutte']
    T = s.pivot_table(index='ricetta', columns='parametro', values='selezionato', aggfunc='sum')
    n = s.groupby('ricetta').id.nunique()
    T = T[[c for c in PARAMETRI_CAL if c in T.columns]]
    T.columns = [NOMI_PARAM[c] for c in T.columns]
    T.insert(0, 'configurazioni', n)
    return T.reset_index()


def anomalie(dati, soglia_chiusura=1e-8, soglia_ident=1e-6):
    R = dati['riepilogo']; righe = []
    for _, r in R.iterrows():
        if pd.notna(r.get('chiusura_max')) and r['chiusura_max'] > soglia_chiusura:
            righe.append(dict(id=r['id'], tipo='chiusura energia', valore=r['chiusura_max']))
        if pd.notna(r.get('A_ident_max')) and r['A_ident_max'] > soglia_ident:
            righe.append(dict(id=r['id'], tipo='identita P=V*I', valore=r['A_ident_max']))
        if pd.notna(r.get('A_sbil_max')) and r['A_sbil_max'] > soglia_ident:
            righe.append(dict(id=r['id'], tipo='sbilancio correnti', valore=r['A_sbil_max']))
    if len(dati['errori']):
        for _, e in dati['errori'].iterrows():
            righe.append(dict(id=e['id'], tipo='errore di esecuzione', valore=str(e['errore'])[:80]))
    return pd.DataFrame(righe)


def report_xlsx(dati, cartella, file='risultati_completi.xlsx'):
    f = os.path.join(cartella, file)
    with pd.ExcelWriter(f, engine='openpyxl') as w:
        dati['riepilogo'].to_excel(w, sheet_name='riepilogo_casi', index=False)
        raccomandazione_globale(dati['riepilogo']).to_excel(w, sheet_name='raccomandazione_globale', index=False)
        for k, nome in (('A', 'A_elettrico'), ('A_conv', 'A_convergenza'), ('B_tempo_fisso', 'B_tempo_fisso'), ('B_conv', 'B_convergenza'),
                        ('B_tau', 'B_tau'), ('C', 'C_temporale'), ('S', 'S_sensibilita')):
            if len(dati[k]):
                dati[k].to_excel(w, sheet_name=nome, index=False)
        t = tabella_parametri_calibrazione(dati)
        if len(t):
            t.to_excel(w, sheet_name='parametri_da_calibrare', index=False)
        fb = frequenza_bloccanti(dati['riepilogo'])
        if len(fb):
            fb.to_excel(w, sheet_name='grandezze_bloccanti', index=False)
        if len(dati.get('E', [])):
            dati['E'].to_excel(w, sheet_name='riferimento_conforme', index=False)
        an = anomalie(dati)
        if len(an):
            an.to_excel(w, sheet_name='anomalie', index=False)
    return f


def report_docx(dati, cartella, opz, grafici, file='report_finale.docx', titolo='Report di sensitivita e verifica numerica del modello ohmico multifase (rev. 2)'):
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    R = dati['riepilogo']
    d = docx.Document()
    for st in d.styles:
        try:
            st.font.name = 'Times New Roman'; st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Times New Roman')
        except Exception:
            pass
    d.styles['Normal'].font.size = Pt(11)
    for sec in d.sections:
        sec.left_margin = sec.right_margin = Cm(2.0); sec.top_margin = sec.bottom_margin = Cm(2.0)
    d.add_heading(titolo, 0)
    d.add_paragraph(f"Data: {pd.Timestamp.now():%d/%m/%Y %H:%M}. Casi analizzati: {len(R)} (ricette: {R.ricetta.nunique()}, configurazioni: {R.config.nunique()}).")

    def tabella(df, colonne, intestazioni=None, larghezze=None, fs=8):
        t = d.add_table(rows=1, cols=len(colonne)); t.style = 'Table Grid'
        for j, c in enumerate(intestazioni or colonne):
            cell = t.rows[0].cells[j]; cell.text = ''; r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(fs)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for j, c in enumerate(colonne):
                v = row[c]
                txt = '-' if (v is None or (isinstance(v, float) and not np.isfinite(v))) else (f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(v))
                cells[j].text = ''; cells[j].paragraphs[0].add_run(txt).font.size = Pt(fs)
        return t

    d.add_heading('1. Impostazioni dell\'analisi', 1)
    voci = [('Soglia di accettazione (scarto dal valore di riferimento)', f"{opz['soglia']:g} %"), ('Criterio di cottura (tau*)', ('attivo, ' + f"{opz['T_target']:g} C") if opz.get('usa_tau') else 'non usato: lettura a tempo fisso'),
            ('Tempo fisso di valutazione', f"{opz['t_fisso']:g} s"), ('Raggio del punto freddo', f"{opz['r_filtro_mm']:g} mm"),
            ('Griglie dello studio A / B', f"{opz['livelli_A']} livelli (rapporto 2) / {opz['griglie_B']} piu' grossolane"),
            ('Passo temporale di base', f"{opz['dt']:g} s"), ("Rumore di misura assunto per l'identificabilita' (PROVVISORIO)", f"{opz['sigma_misura']:g} C"),
            ('Errore standard relativo massimo ammesso', f"{opz['se_max']:g}")]
    t = d.add_table(rows=0, cols=2); t.style = 'Table Grid'
    for a, b in voci:
        c = t.add_row().cells; c[0].text = a; c[1].text = b
    d.add_heading('2. Raccomandazione globale', 1)
    G = raccomandazione_globale(R)
    d.add_paragraph("La dimensione di mesh raccomandata e' la piu' restrittiva fra tutti i casi per cui la soglia e' stata raggiunta con le grandezze essenziali "
                    "(potenza, ripartizione della potenza, temperature e loro differenze, energia). Le grandezze locali (sorgente nei tratti sull'asse, 99° percentile) "
                    "convergono piu' lentamente e sono riportate a parte. Dove la soglia non e' raggiunta con le griglie provate, il caso e' elencato.")
    d.add_paragraph("Per ogni caso viene indicata una dimensione di mesh OPERATIVA: la raccomandata se la soglia e' raggiunta, altrimenti la mesh piu' fine provata, accompagnata dall'incertezza numerica corrispondente (GCI) da riportare con i risultati. "
                    "Se la convergenza e' monotona viene stimata anche la mesh PREVISTA per raggiungere la soglia e il numero di nodi che richiederebbe; e' un'estrapolazione, non una verifica.")
    tabella(G, ['cella', 'casi', 'casi_con_soglia_raggiunta', 'casi_senza_soglia', 'casi_non_asintotici', 'h_raccomandata_mm', 'h_operativa_min_mm', 'incertezza_max_pc', 'dt_raccomandato_s'],
            ['Cella', 'Casi', 'Soglia raggiunta', 'Soglia non raggiunta', 'Non asintotici', 'h raccomandata [mm]', 'h operativa min [mm]', 'Incertezza max [%]', 'dt [s]'], fs=8)
    for _, g in G.iterrows():
        d.add_paragraph(f"{g['cella']}: casi limitanti: {g['casi_limitanti'] or '-'}. Casi senza soglia: {g['casi_senza_soglia_elenco'] or 'nessuno'}.")
    for nome, did in (('G5_mesh.png', 'Dimensione di mesh raccomandata per ricetta e configurazione.'), ('G6_dt.png', 'Passo temporale massimo accettabile.')):
        if nome in grafici:
            d.add_picture(grafici[nome], width=Cm(15.5)); d.add_paragraph(did).alignment = WD_ALIGN_PARAGRAPH.CENTER
    usa_tau = ('stato_tau' in R.columns) and R.stato_tau.notna().any()
    if usa_tau:
        d.add_heading('3. Criterio di cottura: fattibilita\'', 1)
        d.add_paragraph("Per ogni caso si indica se il punto freddo della polpetta raggiunge la temperatura di cottura prima che la purea superi 100 C "
                        "(il modello non include l'ebollizione). Dove e' raggiunto si riporta tau* e la temperatura massima in quell'istante.")
        if 'G4_fattibilita.png' in grafici:
            d.add_picture(grafici['G4_fattibilita.png'], width=Cm(15.5))
    else:
        d.add_heading('3. Entita\' del riscaldamento al tempo di valutazione', 1)
        d.add_paragraph(f"Tutti i risultati sono letti a un tempo fisso ({opz['t_fisso']:g} s, durata delle prove sperimentali); non si impone il raggiungimento di una temperatura obiettivo. "
                        "Le ricette a bassa concentrazione salina, poco conduttive, si scaldano poco in quel tempo: la figura riporta l'incremento massimo di temperatura per ciascun caso. "
                        "Dove l'incremento e' di pochi gradi, uno scarto percentuale sull'incremento diventa severo anche per differenze trascurabili in valore assoluto; "
                        "per questo per le temperature il criterio di accettazione e' la soglia percentuale OPPURE la soglia assoluta "
                        f"({opz.get('tol_abs_C', 0.2):g} C).")
        if 'G4_incremento_T.png' in grafici:
            d.add_picture(grafici['G4_incremento_T.png'], width=Cm(15.5))
    d.add_heading('4. Casi in cui la soglia non e\' raggiunta: grandezza che blocca', 1)
    nd = R[R.h_raccomandata_mm.isna()] if 'h_raccomandata_mm' in R.columns else R.iloc[0:0]
    d.add_paragraph("Quando nessuna delle mesh provate soddisfa la soglia per tutte le grandezze essenziali, non significa che l'incertezza numerica resti sempre sopra la soglia: "
                    "significa che non e' stato dimostrato il contrario con le griglie provate. Per ciascun caso si riportano la mesh piu' fine provata, la grandezza che blocca "
                    "e il suo scarto sulla griglia fine, e l'incertezza GCI massima delle grandezze essenziali. Lo scarto e' accettabile se rispetta la soglia percentuale oppure "
                    "la soglia assoluta sulle temperature.")
    if len(nd):
        tabella(nd.sort_values('id'), ['id', 'h_fine_provata_mm', 'A_bloccante', 'A_scarto_bloccante', 'B_bloccante', 'B_scarto_bloccante', 'A_GCI_ess_max', 'B_GCI_ess_max'],
                ['Caso', 'h piu\' fine [mm]', 'Blocca (A)', 'Scarto A [%]', 'Blocca (B)', 'Scarto B [%]', 'GCI ess. A [%]', 'GCI ess. B [%]'], fs=7)
    else:
        d.add_paragraph("In tutti i casi la soglia e' stata raggiunta.")
    fb = frequenza_bloccanti(R)
    if len(fb):
        d.add_paragraph("Frequenza delle grandezze che bloccano la soglia o che sono le piu' esigenti:")
        tabella(fb, ['studio', 'esito', 'grandezza', 'casi'], ['Studio', 'Esito', 'Grandezza', 'Casi'], fs=8)
    d.add_heading('5. Riepilogo per caso', 1)
    if usa_tau:
        colonne = ['ricetta', 'config', 'A_h_ess_mm', 'B_h_ess_mm', 'C_dt_min', 'h_raccomandata_mm', 'h_fine_provata_mm', 'stato_tau', 'tau_star', 'T_max_a_tau']
        intest = ['Ricetta', 'Config.', 'A h [mm]', 'B h [mm]', 'dt [s]', 'h racc. [mm]', 'h piu\' fine [mm]', 'Cottura', 'tau* [s]', 'Tmax a tau* [C]']
    else:
        colonne = ['ricetta', 'config', 'h_operativa_mm', 'h_stato', 'incertezza_operativa_pc', 'h_temperature_mm', 'h_predetta_mm', 'A_err_app', 'Acorr_h_rec_mm', 'C_dt_min', 't_valutazione', 'dT_max_tfix']
        intest = ['Ricetta', 'Config.', 'h operativa [mm]', 'Stato', 'Incertezza [%]', 'h per temperature [mm]', 'h prevista [mm]', 'Errore A a mesh app [%]', 'h con corr. volume [mm]', 'dt [s]', 'Tempo di valutazione [s]', 'Incremento Tmax [C]']
    col_ok = [c for c in colonne if c in R.columns]
    tabella(R.sort_values(['ricetta', 'config']), col_ok, [intest[colonne.index(c)] for c in col_ok], fs=7.5)
    d.add_heading('6. Convergenza: incertezza numerica per ricetta e configurazione', 1)
    d.add_paragraph("Per ogni ricetta, GCI [%] sulla griglia fine delle grandezze elettriche (in alto) e delle grandezze a tempo fisso (in basso). "
                    "Valori bassi indicano che la mesh e' adeguata per quella grandezza; le grandezze locali (q_asse, q_p99) convergono piu' lentamente.")
    for ric in sorted(R.ricetta.unique()):
        if f'conv:{ric}' in grafici:
            d.add_picture(grafici[f'conv:{ric}'], width=Cm(15.5))
    d.add_heading('7. Parametri consigliati per la calibrazione', 1)
    d.add_paragraph("La sensitivita' e' locale e scalata: variazione di temperatura alle sonde per un aumento del 10% del parametro, a tempi di misura equispaziati. "
                    "I parametri consigliati sono scelti in ordine di sensitivita' purche' tutti i parametri selezionati abbiano errore standard relativo "
                    "non superiore alla soglia, con il rumore di misura assunto. Con sole temperature in uscita sigma e rho*cp tendono a confondersi (il riscaldamento dipende dal loro rapporto).")
    TP = tabella_parametri_calibrazione(dati)
    if len(TP):
        d.add_paragraph("Numero di configurazioni, per ricetta, in cui ciascun parametro e' consigliato (sonde: tutte):")
        tabella(TP, list(TP.columns), fs=7.5)
    for ric in sorted(R.ricetta.unique()):
        if f'sens:{ric}' in grafici:
            d.add_picture(grafici[f'sens:{ric}'], width=Cm(15.5))
    d.add_heading('8. Anomalie e controlli', 1)
    an = anomalie(dati)
    d.add_paragraph("Controlli di coerenza (chiusura dell'energia discreta, identita' P = V I, sbilancio delle correnti) ed errori di esecuzione." if len(an) else
                    "Nessuna anomalia: chiusura dell'energia, identita' P = V I e sbilancio delle correnti entro le soglie; nessun errore di esecuzione.")
    if len(an):
        tabella(an, ['id', 'tipo', 'valore'], fs=8)
    E = dati.get('E')
    if E is not None and len(E):
        d.add_heading('9. Riferimento a interfaccia netta (mesh conforme)', 1)
        d.add_paragraph("Per i casi in cui la soglia non e' stata raggiunta, il problema elettrico e' stato risolto anche su mesh conformi alle sfere (conducibilita' costante per elemento). "
                        "La differenza e' fra la griglia strutturata piu' fine e la mesh conforme piu' fine: indica quanto il valore limite dell'interfaccia diffusa si discosta da quello a interfaccia netta.")
        tabella(E, ['id', 'grandezza', 'strutturata_fine', 'conforme_fine', 'differenza_percento'], ['Caso', 'Grandezza', 'Strutturata fine', 'Conforme fine', 'Differenza [%]'], fs=7.5)
    f = os.path.join(cartella, file)
    d.save(f)
    return f


# =========================================================================================================
# 12. PROTOCOLLO DI INCERTEZZA NUMERICA (mesh strutturata mantenuta) - rev. 6
# =========================================================================================================
# Criterio (fonti nella guida): uncertainty U = Fs * |errore stimato| da raffinamento sistematico (GCI di Celik et al. 2008; versione ai minimi
# quadrati di Eca e Hoekstra 2014 con >= 4 griglie); errore iterativo >= 2-3 ordini sotto quello di discretizzazione e, se non trascurabile, SOMMATO
# ARITMETICAMENTE (la somma in quadratura non e' conservativa: Eca e Hoekstra 2009); componenti spaziale e temporale da studi separati (tre passi sulla
# stessa mesh, tre mesh con lo stesso passo); errore dell'interfaccia diffusa = differenza dai limiti ottenuti con interfaccia netta (mesh conforme).
LC_NETTO = ((8e-3, 4e-3), (5e-3, 2.5e-3), (3.5e-3, 1.75e-3))        # (dimensione globale, dimensione all'interfaccia) [m]

QOI_ARTICOLO = [   # (livello, studio, colonna, etichetta inglese)
    (1, 'B', 'T_P1', 'Temperature at probe P1'), (1, 'B', 'T_P2', 'Temperature at probe P2'), (1, 'B', 'T_P3', 'Temperature at probe P3'),
    (1, 'B', 'T_max', 'Maximum temperature'), (1, 'B', 'scarto_pol_pur', 'Mean temperature difference, puree minus meatballs'),
    (2, 'A', 'P', 'Total electrical power'), (2, 'A', 'frac_P_p', 'Fraction of power dissipated in the meatballs'),
    (2, 'A', 'q_media_polp', 'Mean Joule source in the meatballs'), (2, 'A', 'q_media_pur', 'Mean Joule source in the puree'),
    (2, 'B', 'E_in', 'Electrical energy delivered'),
    (3, 'A', 'q_strati_z', 'Mean Joule source in the puree layers above/below'), (3, 'A', 'q_asse', 'Mean Joule source in the puree gaps on the current axis'),
    (3, 'A', 'q_p99', '99th percentile of the Joule source')]
_T_QOI = ('T_P1', 'T_P2', 'T_P3', 'T_max', 'scarto_pol_pur')


def incertezza_discretizzazione(h, f, modo='auto'):
    """Valore limite e incertezza di discretizzazione su una serie di raffinamento.
    modo='gci3': GCI di Celik sulle TRE mesh piu' fini (ripiego conservativo p = 1, Fs = 3 se la convergenza non e' regolare);
    modo='ls': fit ai minimi quadrati di Eca e Hoekstra su TUTTE le mesh (servono >= 4; con meno si ripiega su 'gci3' e lo si dichiara);
    modo='auto': 'ls' se ci sono >= 4 mesh, altrimenti 'gci3'. Restituisce dict(f_best, U, p, regime, metodo, f_fine)."""
    h = np.asarray(h, float); o = np.argsort(h); h = h[o]; f = np.asarray(f, float)[o]
    out = dict(f_best=np.nan, U=np.nan, p=np.nan, regime='n.d.', metodo='n.d.', f_fine=float(f[0]) if len(f) else np.nan)
    if len(f) < 3 or not np.all(np.isfinite(f)):
        return out
    g = gci_celik(h[:3], f[:3])
    out.update(p=g['p_oss'], regime=g['regime'])
    usa_ls = (modo == 'ls' or modo == 'auto') and len(f) >= 4
    if usa_ls:
        ls = incertezza_ls(h, f)
        out.update(f_best=ls['f0_ls'], U=ls['U_ls'], metodo='least squares (' + ls['ramo'] + ')')
        return out
    note = ' [LS requested, fewer than 4 grids]' if (modo == 'ls' and len(f) < 4) else ''
    if g['regime'] == 'monotona' and np.isfinite(g['p_oss']) and 0.5 <= g['p_oss'] <= 2.5:
        out.update(f_best=g['f_ext'], U=g['GCI_fine'] * abs(f[0]), metodo='GCI (Celik et al.), three finest grids' + note)
    else:
        r = h[1] / h[0]
        out.update(f_best=float(f[0]), U=3.0 * abs(f[0] - f[1]) / (r - 1.0), metodo='fallback p = 1, Fs = 3 (irregular convergence)' + note)
    return out


def _carica_conforme(file_msh):
    import meshio
    m = meshio.read(file_msh)
    mesh = MeshTet(m.points.T, m.cells_dict['tetra'].T.astype(np.int64))
    return mesh, m.cell_data_dict['gmsh:physical']['tetra']


def _t_valutazione(dirc, opz):
    """Tempo di valutazione del caso (salvato nel riepilogo; altrimenti il tempo fisso dell'opzione)."""
    f = os.path.join(dirc, 'riepilogo.json')
    if os.path.exists(f):
        try:
            v = json.load(open(f)).get('t_valutazione')
            if v:
                return float(v)
        except Exception:
            pass
    return opz['t_fisso']


def riferimento_netto_caso(caso, cartella, opz, accoppiato=True, lc_list=LC_NETTO):
    """Famiglia di mesh CONFORMI alle sfere (interfaccia netta) per il caso: problema elettrico a conducibilita' prescritta e, se accoppiato,
    anche le grandezze a tempo fisso. Salva N_elettrico.csv e N_tempo_fisso.csv in cartella/<id>/. Richiede gmsh e meshio."""
    geo, p = caso['geo'], caso['p']
    dirc = os.path.join(cartella, caso['id'])
    if len(geo['centri']) == 0:
        return None
    t_fix = _t_valutazione(dirc, opz)
    rel, rtf = [], []
    for lc, lci in lc_list:
        f = mesh_conforme_gmsh(geo, lc, lci, os.path.join(dirc, 'netto.msh'))
        mesh, tags = _carica_conforme(f)
        ctx = Contesto(geo, None, p, costruisci_filtro=False, mesh_ext=(mesh, tags))
        r = solo_elettrico(ctx, p); r.update(n_nodi=int(ctx.basis.N), n_tet=int(mesh.t.shape[1]), h_rep_mm=ctx.h_rep * 1000, lc_mm=lc * 1e3)
        rel.append(r)
        if accoppiato:
            serie, _ = simula_diag(ctx, p, 1e9, durata_max=t_fix + 2 * opz['dt'], t_min_run=t_fix)
            tf = valori_a_tempo(serie, t_fix); tf.update(n_nodi=int(ctx.basis.N), n_tet=int(mesh.t.shape[1]), h_rep_mm=ctx.h_rep * 1000, lc_mm=lc * 1e3)
            rtf.append(tf)
        del ctx
    for nome in ('netto.msh', 'conforme.msh'):
        try:
            os.remove(os.path.join(dirc, nome))
        except OSError:
            pass
    pd.DataFrame(rel).to_csv(os.path.join(dirc, 'N_elettrico.csv'), index=False)
    if rtf:
        pd.DataFrame(rtf).to_csv(os.path.join(dirc, 'N_tempo_fisso.csv'), index=False)
    return True


def esegui_riferimento_netto(cartella, opz=None, campo_V_cm=4.0, T0=20.0, T_amb=21.0, accoppiato=True, solo_casi=None, verbose=True):
    """Esegue il riferimento a interfaccia netta per i casi gia' calcolati (tutti o l'elenco 'solo_casi' di id). Riprende dai mancanti."""
    o = dict(OPZ_DEFAULT); o.update(opz or {})
    ric = registro_ricette(); base = os.path.join(cartella, 'casi'); n = 0
    for cid in sorted(os.listdir(base)):
        if solo_casi is not None and cid not in solo_casi:
            continue
        if not os.path.exists(os.path.join(base, cid, 'riepilogo.json')):
            continue
        fl = os.path.join(base, cid, 'N_tempo_fisso.csv' if accoppiato else 'N_elettrico.csv')
        if os.path.exists(fl):
            continue
        r_, c_ = cid.split('__')
        try:
            caso = costruisci_caso(r_, c_, ric, campo_V_cm, T0, T_amb, o['dt'], o['durata_max'])
            if riferimento_netto_caso(caso, base, o, accoppiato):
                n += 1
                if verbose:
                    print(f"{cid}: riferimento a interfaccia netta calcolato")
        except Exception as e:
            print(f"{cid}: errore nel riferimento a interfaccia netta: {e}")
    return n


def tabella_incertezza_caso(dirc, opz, T0=20.0, stimatore='gci3'):
    """Tabella dell'incertezza numerica di un caso (formato per l'articolo). Per ogni grandezza: valore sulla mesh di produzione, valore limite stimato,
    incertezza spaziale U_h (griglie strutturate), temporale U_dt (studio sul passo), U_num = U_h + U_dt (somma aritmetica), ed errore dell'interfaccia
    diffusa E_int = (limite strutturato) - (limite a interfaccia netta) con la sua incertezza U_int = U_h + U_h,netto."""
    cid = os.path.basename(dirc.rstrip('/'))
    fA, fB, fC = (os.path.join(dirc, x) for x in ('A.csv', 'B_tempo_fisso.csv', 'C.csv'))
    A = pd.read_csv(fA) if os.path.exists(fA) else pd.DataFrame()
    B = pd.read_csv(fB) if os.path.exists(fB) else pd.DataFrame()
    C = pd.read_csv(fC) if os.path.exists(fC) else pd.DataFrame()
    NA = pd.read_csv(os.path.join(dirc, 'N_elettrico.csv')) if os.path.exists(os.path.join(dirc, 'N_elettrico.csv')) else pd.DataFrame()
    NB = pd.read_csv(os.path.join(dirc, 'N_tempo_fisso.csv')) if os.path.exists(os.path.join(dirc, 'N_tempo_fisso.csv')) else pd.DataFrame()
    senza = cid.endswith('__RP')
    h_prod = opz.get('h_prod_mm', 2.5)
    rise = float(B['T_max'].values[np.argmin(B['h_rep_mm'].values)] - T0) if len(B) and 'T_max' in B.columns else np.nan
    righe = []
    for liv, studio, col, etich in QOI_ARTICOLO:
        if senza and col in ('frac_P_p', 'q_media_polp', 'scarto_pol_pur', 'q_strati_z', 'q_asse'):
            continue
        D, N = (A, NA) if studio == 'A' else (B, NB)
        if D.empty or col not in D.columns:
            continue
        hD = D['h_rep_mm'].values.astype(float)
        est = incertezza_discretizzazione(hD, D[col].values.astype(float), stimatore)
        i_prod = int(np.argmin(np.abs(np.log(hD / h_prod))))
        f_prod = float(D[col].values[i_prod])
        # --- componente temporale (solo per le grandezze dello studio sul passo) ---
        U_dt, f_dt_best, f_dtprod = np.nan, np.nan, np.nan
        if not C.empty and col in C.columns and len(C) >= 3:
            ed = incertezza_discretizzazione(C['dt'].values.astype(float), C[col].values.astype(float), stimatore)
            U_dt, f_dt_best = ed['U'], ed['f_best']
            f_dtprod = float(np.interp(opz.get('dt', 10.0), np.sort(C['dt'].values), C[col].values[np.argsort(C['dt'].values)]))
        scala = rise if col in _T_QOI else abs(est['f_best']) if np.isfinite(est['f_best']) else abs(est['f_fine'])
        U_num = est['U'] + (U_dt if np.isfinite(U_dt) else 0.0)
        r = dict(caso=cid, livello=liv, grandezza=col, etichetta=etich, valore_mesh_produzione=f_prod, valore_griglia_fine=est['f_fine'],
                 valore_limite=est['f_best'], p_osservato=est['p'], regime=est['regime'], metodo=est['metodo'], U_h=est['U'], U_dt=U_dt,
                 U_num=U_num, scala=scala, U_num_rel_pc=100 * U_num / scala if scala else np.nan,
                 errore_mesh_produzione=f_prod - est['f_best'] if np.isfinite(est['f_best']) else np.nan)
        # --- errore dell'interfaccia diffusa rispetto al limite a interfaccia netta ---
        if not N.empty and col in N.columns and len(N) >= 3:
            en = incertezza_discretizzazione(N['h_rep_mm'].values.astype(float), N[col].values.astype(float), 'gci3')
            r.update(limite_interfaccia_netta=en['f_best'], U_h_netta=en['U'], E_int=est['f_best'] - en['f_best'], U_int=est['U'] + en['U'],
                     E_int_rel_pc=100 * (est['f_best'] - en['f_best']) / scala if scala else np.nan, metodo_netta=en['metodo'])
        righe.append(r)
    return pd.DataFrame(righe)


def tabella_incertezza_tutti(cartella, opz=None, T0=20.0, stimatore='gci3'):
    o = dict(OPZ_DEFAULT); o.update(opz or {})
    base = os.path.join(cartella, 'casi'); T = []
    for cid in sorted(os.listdir(base)):
        d = os.path.join(base, cid)
        if os.path.exists(os.path.join(d, 'riepilogo.json')):
            T.append(tabella_incertezza_caso(d, o, T0, stimatore))
    return pd.concat(T, ignore_index=True) if T else pd.DataFrame()


def sintesi_articolo(T):
    """Tabella di sintesi per l'articolo: per ogni grandezza, mediana e massimo (sui casi) dell'incertezza numerica relativa e dell'errore di interfaccia."""
    if T.empty:
        return pd.DataFrame()
    g = T.groupby(['livello', 'grandezza', 'etichetta'], sort=False)
    S = g.agg(casi=('caso', 'nunique'), U_num_rel_mediana_pc=('U_num_rel_pc', 'median'), U_num_rel_max_pc=('U_num_rel_pc', 'max'),
              irregolari=('regime', lambda x: int((x != 'monotona').sum()))).reset_index()
    if 'E_int_rel_pc' in T.columns:
        E = g['E_int_rel_pc'].agg(lambda x: np.nanmedian(np.abs(x)) if x.notna().any() else np.nan).rename('E_int_mediana_pc').reset_index()
        Em = g['E_int_rel_pc'].agg(lambda x: np.nanmax(np.abs(x)) if x.notna().any() else np.nan).rename('E_int_max_pc').reset_index()
        S = S.merge(E, on=['livello', 'grandezza', 'etichetta']).merge(Em, on=['livello', 'grandezza', 'etichetta'])
    return S


def verifica_stimatori_caso(caso, opz, h_base_mm=None, max_nodi=140000, verbose=True, t_val=None):
    """Verifica degli stimatori di incertezza con una mesh PIU' FINE di quelle usate per stimare (metodologia di Eca et al., 2021: errore 'esatto' da griglie
    piu' raffinate). Famiglia di 5 mesh a rapporto 2: le prime 4 servono a stimare (GCI sulle 3 piu' fini e minimi quadrati sulle 4), la quinta e' il riferimento.
    Errore di riferimento: e_ref = f(mesh 4) - f(mesh 5); errore stimato rispetto al limite: e_vero = e_ref / (1 - 2^-p) con p osservato sulle tre mesh piu' fini
    (p = 1 se la convergenza non e' monotona). Una stima 'copre' se U >= |e_vero|. Restituisce un DataFrame per grandezza (None se la cella non ha 5 mesh)."""
    geo, p = caso['geo'], caso['p']
    if len(geo['centri']) == 0:
        return None
    G = griglie_per_geo(geo, h_base_mm or 2 * opz['h_base_mm'], 5, max_nodi)
    if len(G) < 5:
        return None
    t_fix = (t_val if t_val else opz['t_fisso'])
    rA, rB = [], []
    for n in G:
        ctx = Contesto(geo, n, p, costruisci_filtro=False)
        e = solo_elettrico(ctx, p); e.update(h_rep_mm=ctx.h_rep * 1000, n_nodi=int(ctx.basis.N)); rA.append(e)
        serie, _ = simula_diag(ctx, p, 1e9, durata_max=t_fix + 2 * opz['dt'], t_min_run=t_fix)
        tf = valori_a_tempo(serie, t_fix); tf.update(h_rep_mm=ctx.h_rep * 1000, n_nodi=int(ctx.basis.N)); rB.append(tf)
        if verbose:
            print(f"   {caso['id']}: mesh {n} ({ctx.basis.N} nodi) fatta", flush=True)
        del ctx
    A, B = pd.DataFrame(rA), pd.DataFrame(rB)
    rise = float(B['T_max'].values[-1] - p['T_iniziale'])
    righe = []
    for liv, studio, col, etich in QOI_ARTICOLO:
        D = A if studio == 'A' else B
        if col not in D.columns:
            continue
        h = D['h_rep_mm'].values.astype(float); f = D[col].values.astype(float)
        eg = incertezza_discretizzazione(h[:4], f[:4], 'gci3'); el = incertezza_discretizzazione(h[:4], f[:4], 'ls')
        e_ref = f[3] - f[4]
        g5 = gci_celik(h[2:5][::-1][::-1], f[2:5]) if False else gci_celik(np.sort(h)[:3], f[np.argsort(h)][:3])   # tre mesh piu' fini della famiglia completa (la finestra 3,4,5 e' la piu' fine)
        pr = float(np.clip(g5['p_oss'], 0.5, 2.0)) if (g5['regime'] == 'monotona' and np.isfinite(g5['p_oss'])) else 1.0
        e_vero = abs(e_ref) / (1.0 - 2.0 ** (-pr))
        scala = rise if col in _T_QOI else abs(f[4])
        righe.append(dict(caso=caso['id'], livello=liv, grandezza=col, etichetta=etich, valore_riferimento=f[4], errore_mesh4=e_ref, errore_stimato_vero=e_vero, ordine_riferimento=pr,
                          U_GCI3=eg['U'], U_LS4=el['U'], scala=scala,
                          rapporto_GCI3=eg['U'] / e_vero if e_vero > 0 else np.nan, rapporto_LS4=el['U'] / e_vero if e_vero > 0 else np.nan,
                          copre_GCI3=bool(eg['U'] >= e_vero), copre_LS4=bool(el['U'] >= e_vero),
                          err_limite_GCI3=abs(eg['f_best'] - f[4]), err_limite_LS4=abs(el['f_best'] - f[4]),
                          regime_GCI3=eg['regime'], metodo_LS4=el['metodo']))
    return pd.DataFrame(righe)


def sintesi_verifica(V):
    """Per stimatore: percentuale di grandezze in cui l'incertezza copre l'errore vero e rapporto mediano U / errore vero."""
    if V is None or V.empty:
        return pd.DataFrame()
    righe = []
    for livello, g in list(V.groupby('livello')) + [('tutti', V)]:
        for nome, c_cop, c_rap in (('GCI sulle 3 mesh piu\' fini', 'copre_GCI3', 'rapporto_GCI3'), ('Minimi quadrati su 4 mesh', 'copre_LS4', 'rapporto_LS4')):
            righe.append(dict(livello=livello, stimatore=nome, grandezze=len(g), copertura_pc=100 * float(g[c_cop].mean()),
                              rapporto_mediano=float(np.nanmedian(g[c_rap])), rapporto_p10=float(np.nanpercentile(g[c_rap], 10)), rapporto_p90=float(np.nanpercentile(g[c_rap], 90))))
    return pd.DataFrame(righe)


def tabelle_articolo_docx(tabelle, file='tabelle_articolo.docx', validazione=None):
    """Word con le tabelle per il manoscritto (solo inglese, Times New Roman). 'tabelle' = lista di (titolo breve, T, S): una coppia (sintesi + completa) per ogni stimatore.
    'validazione' = (V, sintesi) della verifica con mesh piu' fine. Le note di lavoro stanno nei fogli Excel, non qui."""
    import docx
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    d = docx.Document()
    for st in d.styles:
        try:
            st.font.name = 'Times New Roman'; st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Times New Roman')
        except Exception:
            pass
    for sec in d.sections:
        sec.left_margin = sec.right_margin = Cm(1.8); sec.top_margin = sec.bottom_margin = Cm(2.0)
    liv = {1: 'Level 1 (temperatures)', 2: 'Level 2 (power and energy)', 3: 'Level 3 (local sources, diagnostic)'}

    def tab(df, cols, heads, fs=8):
        t = d.add_table(rows=1, cols=len(cols)); t.style = 'Table Grid'
        for j, h in enumerate(heads):
            c = t.rows[0].cells[j]; c.text = ''; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(fs)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for j, c in enumerate(cols):
                v = row[c]
                txt = '-' if (v is None or (isinstance(v, float) and not np.isfinite(v))) else (f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(v))
                cells[j].text = ''; cells[j].paragraphs[0].add_run(txt).font.size = Pt(fs)
    letter = 'A'
    for titolo, T, S in tabelle:
        d.add_paragraph().add_run(f'Table {letter}. Numerical uncertainty of the simulated quantities across all recipes and configurations: {titolo}.').bold = True
        d.add_paragraph('Relative numerical uncertainty U_num = U_h + U_dt (arithmetic sum of the spatial and temporal components) is expressed as a percentage of the temperature increase '
                        '(temperatures) or of the quantity itself (other quantities). E_int is the difference between the limit of the diffuse-interface structured-grid sequence and the limit '
                        'obtained with interface-fitted meshes, as a percentage of the same scale. Median and maximum are taken over the cases.')
        if not S.empty:
            S2 = S.copy(); S2['Level'] = S2['livello'].map(liv)
            cols = ['Level', 'etichetta', 'casi', 'U_num_rel_mediana_pc', 'U_num_rel_max_pc', 'irregolari'] + [c for c in ('E_int_mediana_pc', 'E_int_max_pc') if c in S2.columns]
            heads = ['Level', 'Quantity', 'Cases', 'U_num median [%]', 'U_num max [%]', 'Non-asymptotic cases'] + [h for c, h in (('E_int_mediana_pc', 'E_int median [%]'), ('E_int_max_pc', 'E_int max [%]')) if c in S2.columns]
            tab(S2, cols, heads)
        d.add_page_break()
        d.add_paragraph().add_run(f'Table {letter}-S (Supplementary). Numerical uncertainty by case and quantity: {titolo}.').bold = True
        if not T.empty:
            cols = [c for c in ['caso', 'grandezza', 'valore_mesh_produzione', 'valore_limite', 'p_osservato', 'regime', 'U_h', 'U_dt', 'U_num_rel_pc', 'E_int', 'U_int'] if c in T.columns]
            heads = {'caso': 'Case', 'grandezza': 'Quantity', 'valore_mesh_produzione': 'Value (production grid)', 'valore_limite': 'Estimated limit', 'p_osservato': 'Observed order',
                     'regime': 'Convergence', 'U_h': 'U_h', 'U_dt': 'U_dt', 'U_num_rel_pc': 'U_num [%]', 'E_int': 'E_int', 'U_int': 'U_int'}
            tab(T, cols, [heads[c] for c in cols], fs=7)
        d.add_page_break()
        letter = chr(ord(letter) + 1)
    if validazione is not None:
        V, SV = validazione
        d.add_paragraph().add_run(f'Table {letter}. Assessment of the two discretization-uncertainty estimators against a finer grid.').bold = True
        d.add_paragraph('The estimators were applied to the four coarser grids of a five-grid family (refinement ratio two); the finest grid served as reference. A stated uncertainty is said to cover the error '
                        'when it is not smaller than the error of the fourth grid extrapolated to the limit. Ratio = uncertainty divided by that error (median, 10th and 90th percentile over the quantities).')
        if not SV.empty:
            tab(SV, ['livello', 'stimatore', 'grandezze', 'copertura_pc', 'rapporto_mediano', 'rapporto_p10', 'rapporto_p90'],
                ['Level', 'Estimator', 'Quantities', 'Coverage [%]', 'Median ratio', 'Ratio P10', 'Ratio P90'])
    d.save(file)
    return file
