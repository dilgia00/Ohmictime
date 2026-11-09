"""Interfaccia con il motore FEM del protocollo per la cella 99 x 54 x 66 mm: costruzione dei casi, simulazioni, campi di temperatura e analisi dirette."""
import os, sys, threading, tempfile, time
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import LinearNDInterpolator

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ENG = os.path.join(AQUI, 'engine')
if ENG not in sys.path: sys.path.insert(0, ENG)
import logging; logging.getLogger('skfem').setLevel(logging.ERROR)
import diagnostica_ohmico as dg
import multifase_engine as me
import protocollo_articolo as pa
from . import dati

MESHDIR = os.path.join(tempfile.gettempdir(), 'mesh_ohmico'); os.makedirs(MESHDIR, exist_ok=True)
LOCK = threading.Lock()
NODI_LIVELLO = [800, 2263, 6400, 18102, 51200]
try:
    import gmsh; GMSH = True
except Exception:
    GMSH = False
FAMIGLIE = {'strutturata': 'Griglia strutturata (interfaccia diffusa)', 'conforme': 'Mesh conforme (interfaccia netta)'}
NOM = dict(sig_m20=None, alfa_m=None, sig_p20=None, alfa_p=None)


def famiglia_default(): return 'conforme' if GMSH else 'strutturata'


@st.cache_resource(show_spinner=False)
def contesto(config, famiglia, livello):
    """Contesto di calcolo (mesh, spazi di elementi, operatori): dipende dalla configurazione e dalla mesh, non dalla ricetta."""
    caso = dg.costruisci_caso('A_40V', config); o = pa.opzioni_art(dict(cartella_mesh=MESHDIR))
    with LOCK: return pa.contesto(caso, famiglia, int(livello), o)


def nominali(ric, V, config):
    vk = int(V) if int(V) in (30, 40, 50) else 40            # il registro dei casi ha 30, 40 e 50 V: le altre tensioni sono imposte dopo (V_rms)
    caso = dg.costruisci_caso(f'{ric}_{vk}V', config); p = caso['p']
    return dict(sig_m20=p['_sig_m20'], alfa_m=p['_alfa_m'], sig_p20=p['_sig_p20'], alfa_p=p['_alfa_p'], fk_m=1.0, fk_p=1.0, fc_m=1.0, fc_p=1.0, fh=1.0), caso


def parametri(ric, V, config, T0=20.0, T0_sfere=None, T_amb=21.0, dt=10.0, durata=270.0, mult=None, assoluti=None):
    """Dizionario di parametri del motore. 'mult': moltiplicatori dei nove parametri; 'assoluti': sostituzioni dirette (es. V_rms)."""
    nom, caso = nominali(ric, V, config); p = dict(caso['p']); mult = mult or {}
    th = {k: nom[k] * float(mult.get(k, 1.0)) for k in nom}; p = dg.applica_param(p, th)
    p.update(T_iniziale=float(T0), T_iniziale_polpetta=float(np.mean(T0_sfere)) if T0_sfere is not None else float(T0), T_amb=float(T_amb), dt=float(dt), durata=float(durata), V_rms=float(V))
    if T0_sfere is not None: p['T_iniziale_sfere'] = [float(x) for x in T0_sfere]
    if assoluti: p.update(assoluti)
    return p, caso


def simula(ric, V, config, famiglia, livello, T0=20.0, T0_sfere=None, T_amb=21.0, dt=10.0, durata=270.0, mult=None, assoluti=None, snap_n=0, avanzamento=None):
    """Una simulazione completa. Restituisce dict(serie, info, ctx, p, tempo)."""
    ctx = contesto(config, famiglia, livello); p, caso = parametri(ric, V, config, T0, T0_sfere, T_amb, dt, durata, mult, assoluti); t0 = time.time()
    with LOCK: serie, info = dg.simula_diag(ctx, p, 1e9, durata_max=float(durata), t_min_run=float(durata), snap_n=snap_n, avanzamento=avanzamento)
    return dict(serie=serie, info=info, ctx=ctx, p=p, tempo=time.time() - t0, caso=caso)


def stima_costo(config, livello): return {0: 0.8, 1: 3.0, 2: 10.0, 3: 38.0, 4: 150.0}.get(int(livello), 10.0)


def stato_famiglia(famiglia, livello):
    if famiglia == 'conforme' and not GMSH: return False, 'La mesh conforme richiede gmsh, non disponibile in questo ambiente: si usa la griglia strutturata.'
    return True, ''


_TRI = {}


def triangolazione(ctx):
    """Triangolazione di Delaunay dei nodi, costruita una volta per contesto (serve solo alla visualizzazione dei tagli)."""
    from scipy.spatial import Delaunay
    k = id(ctx)
    if k not in _TRI: _TRI[k] = Delaunay(np.asarray(ctx.mesh.p).T)
    return _TRI[k]


def fetta(ctx, campo, asse, pos_mm, nu=140, nv=90):
    """Valori del campo nodale su un piano di taglio. asse: 'x', 'y' o 'z'; pos in mm. Restituisce (u, v, Z) in mm."""
    tri = triangolazione(ctx); L = [ctx.geo['Lx'], ctx.geo['Ly'], ctx.geo['Lz']]; ax = 'xyz'.index(asse); idx = [i for i in range(3) if i != ax]
    u = np.linspace(0, L[idx[0]], nu); v = np.linspace(0, L[idx[1]], nv); U, V = np.meshgrid(u, v); pts = np.zeros((U.size, 3)); pts[:, idx[0]] = U.ravel(); pts[:, idx[1]] = V.ravel(); pts[:, ax] = pos_mm / 1000.0
    Z = LinearNDInterpolator(tri, np.asarray(campo, float))(pts).reshape(U.shape)
    return u * 1000, v * 1000, Z


def sigma_nodale(ctx, T, p): return np.asarray(me.sigma_campo(np.asarray(T, float), np.asarray(ctx.phi_n, float), p), float)


def indici_confronto(S, D):
    S, D = np.asarray(S), np.asarray(D); e = S - D
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)), MAE=float(np.mean(np.abs(e))), R2=float(1 - np.sum(e ** 2) / max(np.sum((D - D.mean()) ** 2), 1e-12)), n=int(len(D)))
