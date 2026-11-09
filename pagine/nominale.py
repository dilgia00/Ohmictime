"""Modello con parametri nominali contro tutte le misure: verifica preliminare prima della calibrazione."""
import time
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import ui, dati, calibra
from core import motore as mo
import calibrazione_protocollo as cp

ui.applica_tema()
ui.hero("Modello nominale contro le misure", "Verifica preliminare, prima di qualsiasi calibrazione: il gemello digitale con i parametri nominali (conducibilità misurate, proprietà di Choi-Okos) è confrontato con gli incrementi di temperatura delle prove, condizione per condizione, per vedere dove sovrastima o sottostima il riscaldamento e quanto è lontano dal rumore di misura.", ["incrementi ΔT(t) alle sonde", "bias per ricetta, tensione e configurazione", "confronto con u_D"])
S = st.session_state; ud = calibra.uD_stima(1.5); G_all = calibra.gruppi_lista(2.0); fams = ['strutturata'] + (['conforme'] if mo.GMSH else [])


@st.cache_resource(show_spinner=False)
def simulatore(fam, liv, dt, Ta): return calibra.nuovo_simulatore(fam, liv, dt, Ta)


ui.sezione("Selezione delle condizioni")
c = st.columns(3); fr = c[0].multiselect("Ricette", dati.RICS, default=dati.RICS, format_func=lambda x: f"{x} · {dati.SALE[x]:g}%", key="no_r"); fv = c[1].multiselect("Tensioni (V)", list(dati.VS), default=list(dati.VS), key="no_v"); fc = c[2].multiselect("Configurazioni", dati.CFG, default=dati.CFG, key="no_c")
G = [g for g in G_all if g['ric'] in fr and int(g['V']) in fv and g['cfg'] in fc]; c = st.columns([1.4, 1, 1, 1, 1]); fam = c[0].selectbox("Mesh", fams, index=len(fams) - 1, format_func=lambda f: mo.FAMIGLIE[f], key="no_f"); liv = c[1].select_slider("Livello", [0, 1], value=0, format_func=lambda k: f"{k} · {mo.NODI_LIVELLO[k]:,} nodi".replace(',', ' '), key="no_l"); dt = c[2].select_slider("Passo (s)", [10.0, 20.0, 30.0], value=20.0, key="no_dt")
Ta = c[3].number_input("T aria (°C)", 15.0, 30.0, 21.0, 0.5, key="no_ta"); nmax = c[4].slider("Gruppi (massimo)", 6, max(len(G), 7), min(len(G), 40) if len(G) > 6 else 6, key="no_n")
rng = np.random.default_rng(11); sel = [G[i] for i in sorted(rng.choice(len(G), min(nmax, len(G)), replace=False))] if G else []; tsim = {0: 0.5, 1: 2.0}[liv] * (10.0 / dt)
k = st.columns(4); ui.kpi(k[0], f"{len(G_all)}", "gruppi di condizioni nei dati", f"{len(calibra.prove_lista())} prove"); ui.kpi(k[1], f"{len(G)}", "gruppi selezionati", f"{sum(len(g['runs']) for g in G)} prove"); ui.kpi(k[2], f"{len(sel)}", "gruppi da simulare", f"tempo stimato {len(sel) * tsim / 60:.1f} min"); ui.kpi(k[3], f"{ud['a']:.3f} · {ud['b']:.2f} K", "rumore di misura a · b", "u_D = √((aΔT)² + b²)")
if st.button("▶ Confronta il modello nominale con le misure", type="primary", disabled=not sel, key="no_go"):
    sim = simulatore(fam, liv, dt, Ta); barra = st.progress(0.0, text="Simulazioni…"); sims = []; t0 = time.time()
    for i, g in enumerate(sel): sims.append(sim.calcola([g], {})[0]); barra.progress((i + 1) / len(sel), text=f"Gruppo {i + 1} di {len(sel)}")
    Sv, Dv, Uv, M = cp.coppie(sel, sims, ud); P = M.assign(S=Sv, D=Dv, U=Uv); P['res'] = P.S - P.D; P['z'] = P.res / P.U; barra.empty(); S.nominale = dict(P=P, fam=fam, liv=liv, dt=dt, tempo=time.time() - t0, ngr=len(sel))
N = S.get('nominale')
if N is None: ui.callout("Premi il pulsante per confrontare il modello nominale con le misure. A mesh grossolana (livello 0) ogni gruppo richiede circa mezzo secondo.")
else:
    P = N['P']; ind = mo.indici_confronto(P.S, P.D); rum = float(np.sqrt(np.mean(P.U ** 2))); ui.sezione("Prestazioni del modello nominale"); k = st.columns(5)
    ui.kpi(k[0], f"{ind['RMSE']:.2f} K", "RMSE", f"rumore di misura rms {rum:.2f} K"); ui.kpi(k[1], f"{ind['bias']:+.2f} K", "bias medio", "simulato − misurato"); ui.kpi(k[2], f"{ind['MAE']:.2f} K", "MAE", ""); ui.kpi(k[3], f"{ind['R2']:.3f}", "R²", "sugli incrementi"); ui.kpi(k[4], f"{100 * (P.res.abs() <= P.U).mean():.0f}% · {100 * (P.res.abs() <= 2 * P.U).mean():.0f}%", "entro u_D · entro 2 u_D", f"{ind['n']} punti, {N['ngr']} gruppi")
    if ind['bias'] > 0.1 * ind['RMSE']: ui.callout(f"Il modello nominale <b>sovrastima</b> il riscaldamento (bias {ind['bias']:+.2f} K): è l'effetto atteso di conducibilità e capacità termiche non calibrate, ed è il punto di partenza per la calibrazione di k_σ e f_c,m.", 'warn')
    elif ind['bias'] < -0.1 * ind['RMSE']: ui.callout(f"Il modello nominale <b>sottostima</b> il riscaldamento (bias {ind['bias']:+.2f} K).", 'warn')
    a, b = st.columns(2); Bv = P.groupby(['ric', 'V']).res.mean().unstack().reindex(dati.RICS).dropna(how='all'); fig = go.Figure(go.Heatmap(z=Bv.values, x=[f"{v:g} V" for v in Bv.columns], y=Bv.index, colorscale='RdBu_r', zmid=0, text=np.round(Bv.values, 2), texttemplate='%{text}', colorbar=dict(title='K')))
    ui.layout(fig, "Bias per ricetta e tensione (simulato − misurato, K)", 400, yaxis=dict(autorange='reversed')); a.plotly_chart(fig, width='stretch'); Rv = P.groupby(['ric', 'cfg']).res.apply(lambda x: float(np.sqrt(np.mean(x ** 2)))).unstack().reindex(dati.RICS).reindex(columns=dati.CFG).dropna(how='all'); fig = go.Figure(go.Heatmap(z=Rv.values, x=Rv.columns, y=Rv.index, colorscale='YlOrRd', text=np.round(Rv.values, 2), texttemplate='%{text}', colorbar=dict(title='K')))
    ui.layout(fig, "RMSE per ricetta e configurazione (K)", 400, yaxis=dict(autorange='reversed')); b.plotly_chart(fig, width='stretch'); a, b = st.columns(2); fig = go.Figure(); cm = dict(zip(dati.RICS, ui.COL_RIC))
    for r_ in dati.RICS:
        x = P[P.ric == r_]
        if len(x): fig.add_trace(go.Scatter(x=x.D, y=x.S, mode='markers', marker=dict(size=5, color=cm[r_], opacity=0.55), name=f"{r_} · {dati.SALE[r_]:g}%"))
    lim = [0, float(max(P.D.max(), P.S.max()))]; fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines', line=dict(color='black', dash='dash'), name='identità')); ui.layout(fig, "Simulato contro misurato (incrementi, K)", 420, xaxis_title='Misurato (K)', yaxis_title='Simulato (K)'); a.plotly_chart(fig, width='stretch')
    fig = go.Figure()
    for ph, col in (('polpetta', ui.COL['conf']), ('purea', ui.COL['strutt'])):
        x = P[P.fase == ph]
        if len(x): g = x.groupby('t').res.mean().reset_index(); fig.add_trace(go.Scatter(x=g.t, y=g.res, mode='lines', line=dict(color=col, width=3), name=f"sonde in {ph}"))
    fig.add_hline(y=0, line=dict(color='black', width=1)); ui.layout(fig, "Bias nel tempo per fase della sonda (K)", 420, xaxis_title='Tempo (s)', yaxis_title='K', legend=dict(orientation='h', y=-0.2)); b.plotly_chart(fig, width='stretch')
    ui.sezione("Condizioni con il maggiore scarto"); W = P.groupby('cond').agg(RMSE=('res', lambda x: float(np.sqrt(np.mean(x ** 2)))), bias=('res', 'mean'), u_rms=('U', lambda x: float(np.sqrt(np.mean(x ** 2)))), punti=('res', 'size')).reset_index(); W['RMSE / u_D'] = W.RMSE / W.u_rms; st.dataframe(W.sort_values('RMSE', ascending=False).head(12).round(2).rename(columns={'cond': 'Condizione'}), hide_index=True, width='stretch')
    st.download_button("Scarica i confronti (CSV)", P.to_csv(index=False), "modello_nominale_vs_misure.csv", "text/csv", key="no_dl")
ui.piede()
