"""Cella, ricette e dati sperimentali."""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import ui, dati, calibra
from core import motore as mo
import calibrazione_protocollo as _cp

cp_sfere = lambda r: _cp.SFERE[r['cfg']]

ui.applica_tema()
ui.hero("Cella, ricette e dati sperimentali", "Geometria della cella con le cinque configurazioni delle polpette, conducibilità delle ricette e delle polpette, e analisi dei dati sperimentali con il modello del rumore di misura.", ["99 × 54 × 66 mm", "sfere R = 15 mm", "termocoppie tipo T", "256 prove"])
tab_c, tab_r, tab_d = st.tabs(["Cella e configurazioni", "Ricette e conducibilità", "Dati sperimentali"])

with tab_c:
    a, b = st.columns([1.5, 1])
    conf = a.radio("Configurazione", dati.CFG, horizontal=True, format_func=lambda c: f"{c} ({dati.NPOLP[c]} polpette)", key="cd_conf")
    L, W, H, R = 99.0, 54.0, 66.0, 15.0; xs = [17.25, 49.5, 81.75]; fig = go.Figure()
    e = [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0), (0, 0, 0), (0, 0, H), (L, 0, H), (L, W, H), (0, W, H), (0, 0, H)]
    fig.add_trace(go.Scatter3d(x=[p[0] for p in e], y=[p[1] for p in e], z=[p[2] for p in e], mode='lines', line=dict(color='#343A40', width=4), showlegend=False, hoverinfo='skip'))
    for p0, p1 in (((L, 0, 0), (L, 0, H)), ((L, W, 0), (L, W, H)), ((0, W, 0), (0, W, H))): fig.add_trace(go.Scatter3d(x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]], mode='lines', line=dict(color='#343A40', width=4), showlegend=False, hoverinfo='skip'))
    for x0, nm in ((0, 'Elettrodo 1 (ψ = 1)'), (L, 'Elettrodo 2 (ψ = 0)')): fig.add_trace(go.Mesh3d(x=[x0] * 4, y=[0, W, W, 0], z=[0, 0, H, H], i=[0, 0], j=[1, 2], k=[2, 3], color='#868E96', opacity=0.55, name=nm, showlegend=True, hovertemplate=nm + '<extra></extra>'))
    th, ph = np.meshgrid(np.linspace(0, np.pi, 28), np.linspace(0, 2 * np.pi, 48), indexing='ij')
    for i, x in enumerate(xs):
        occ = dati.OCC[conf][i] == 1
        if occ: fig.add_trace(go.Surface(x=x + R * np.sin(th) * np.cos(ph), y=W / 2 + R * np.sin(th) * np.sin(ph), z=H / 2 + R * np.cos(th), colorscale=[[0, '#E8590C'], [1, '#F08C00']], showscale=False, opacity=0.85, name=f'Polpetta {i + 1}', showlegend=False, hovertemplate=f'Polpetta {i + 1}<extra></extra>'))
        fig.add_trace(go.Scatter3d(x=[x], y=[W / 2], z=[H / 2], mode='markers+text', marker=dict(size=6, color='#0B3954'), text=[f'P{i + 1}'], textposition='top center', name=f"P{i + 1} ({'polpetta' if occ else 'purea'})", showlegend=True))
    fig.update_layout(height=520, scene=dict(xaxis=dict(title='x (mm)', range=[-5, L + 5]), yaxis=dict(title='y (mm)', range=[-5, W + 5]), zaxis=dict(title='z (mm)', range=[-5, H + 5]), aspectmode='data', camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9))), margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation='h', y=0))
    a.plotly_chart(fig, width='stretch')
    with b:
        st.markdown("**Geometria**"); st.markdown("- Dominio: **99 × 54 × 66 mm** (volume 352.8 cm³): L fra gli elettrodi, che coprono le facce estreme\n- Sfere: diametro 30 mm, **equidistanti** (spazio di 2.25 mm fra le sfere e con le pareti)\n- Centri: x = 17.25, 49.5, 81.75 mm; y = 27 mm; z = 33 mm\n- Superficie libera in alto (scambio per convezione naturale e irraggiamento)")
        t = pd.DataFrame({'Posizione': ['1', '2', '3'], 'x (mm)': xs, 'Occupata': ['sì' if o else 'no' for o in dati.OCC[conf]], 'Sonda': [f'P{i + 1}' for i in range(3)], 'Fase alla sonda': ['polpetta' if o else 'purea' for o in dati.OCC[conf]]})
        st.dataframe(t, hide_index=True, width='stretch')
        ui.callout("Il numero di polpette (3, 2, 2, 1, 1) è dedotto dalle sonde in purea disponibili in ciascuna configurazione. <b>Tris</b> non ha sonde in purea.", 'warn')

with tab_r:
    T = np.linspace(10, 90, 161); fig = make_subplots(rows=1, cols=2, subplot_titles=("Conducibilità elettrica σ(T)", "Contrasto σ_p / σ_m in funzione della temperatura"), horizontal_spacing=0.09)
    for r, col in zip(dati.RICS, ui.COL_RIC):
        fig.add_trace(go.Scatter(x=T, y=dati.sigma_T(r, T), name=f"{r} ({dati.SALE[r]:g}% sale)", line=dict(color=col, width=2.6), legendgroup=r), row=1, col=1)
        fig.add_trace(go.Scatter(x=T, y=dati.sigma_polp(T) / dati.sigma_T(r, T), name=r, line=dict(color=col, width=2.6), legendgroup=r, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=T, y=dati.sigma_polp(T), name='Polpette', line=dict(color='#212529', width=3, dash='dash')), row=1, col=1); fig.add_hline(y=1, line=dict(color='#868E96', dash='dot'), row=1, col=2)
    fig.update_xaxes(title='Temperatura (°C)'); fig.update_yaxes(title='σ (S/m)', row=1, col=1); fig.update_yaxes(title='Contrasto (-)', type='log', row=1, col=2); fig.update_layout(height=430, legend=dict(orientation='h', y=-0.2)); st.plotly_chart(fig, width='stretch')
    st.caption("Il contrasto cambia durante il riscaldamento, perché purea e polpette hanno coefficienti di temperatura diversi: per le ricette con poco sale, ricche di acqua e poco conduttive, il coefficiente relativo della purea è più alto.")
    rows = []
    for r in dati.RICS:
        a_, b_ = dati.AB[r]; s20 = 20 * a_ + b_; ric = mo.me.RICETTE_SPERIMENTALI[r]; m = ric['masse']; tot = sum(m.values())
        rows.append({'Ricetta': r, 'Sale (%)': dati.SALE[r], 'a (S/m/°C)': a_, 'b (S/m)': b_, 'σ₂₀ (S/m)': round(s20, 3), 'α (1/K)': round(a_ / s20, 4), 'Contrasto a 20 °C': round(dati.CONTR[r], 2), 'Acqua (% w/w)': round(100 * m['acqua'] / tot, 1), 'Fiocchi (% w/w)': round(100 * m['fiocchi'] / tot, 1), 'Burro (% w/w)': round(100 * m['burro'] / tot, 1), 'Sale (% w/w)': round(100 * m['sale'] / tot, 2), 'Fonte σ(T)': f"{ric['sigma'][2]} ({ric['sigma'][3]})"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    ui.callout("Le conducibilità delle ricette A, B, D ed E sono ricavate dal grafico delle misure (16 marcatori digitalizzati, R² > 0.9998). La ricetta <b>C</b> (0% di sale) non compare nel grafico e usa la regressione della Tabella 6.5 della tesi: da confermare. Le composizioni sono calcolate sulla somma dei componenti; il totale riportato nella tabella delle masse differisce (per la ricetta A, 617.58 g contro 599.94 g).", 'warn')

with tab_d:
    X = dati.sperimentali(); P = calibra.catalogo_prove()
    c = st.columns(4); f_r = c[0].multiselect("Ricetta", dati.RICS, default=['E'], key="cd_fr"); f_v = c[1].multiselect("Tensione (V)", list(dati.VS), default=[40], key="cd_fv"); f_c = c[2].multiselect("Configurazione", dati.CFG, default=['Tris'], key="cd_fc"); modo = c[3].radio("Grandezza", ["Incremento ΔT", "Temperatura"], horizontal=True, key="cd_mod")
    sel = P[P.ric.isin(f_r) & P.V.isin(f_v) & P.cfg.isin(f_c)]
    if sel.empty: st.info("Nessuna prova con questa combinazione: modifica i filtri.")
    else:
        k = st.columns(4); ui.kpi(k[0], f"{len(sel)}", "prove selezionate"); ui.kpi(k[1], f"{sel.groupby(['ric', 'V', 'cfg']).ngroups}", "condizioni"); ui.kpi(k[2], f"{sel.T0_purea.min():.1f}–{sel.T0_purea.max():.1f} °C", "temperatura iniziale della purea"); ui.kpi(k[3], f"{sel[['dT_P1', 'dT_P2', 'dT_P3']].max().max():.1f} K", "massimo incremento finale")
        fig = make_subplots(rows=1, cols=3, subplot_titles=("P1", "P2", "P3"), shared_yaxes=True, horizontal_spacing=0.03)
        for run in sel.run:
            r = calibra.prova(run)
            for p in range(3):
                y = r['dT'][:, p] if modo.startswith('Inc') else r['dT'][:, p] + (r['T0m'] if p not in cp_sfere(r) else r['T0s'][cp_sfere(r).index(p)])
                fig.add_trace(go.Scatter(x=r['t'], y=y, mode='lines', line=dict(width=1.3), name=f"prova {run} ({r['ric']}, {int(r['V'])} V, {r['cfg']})", legendgroup=str(run), showlegend=(p == 0), hovertemplate='t=%{x:.0f} s<br>%{y:.2f}<extra>prova ' + str(run) + '</extra>'), row=1, col=p + 1)
        fig.update_xaxes(title='Tempo (s)'); fig.update_yaxes(title='ΔT (K)' if modo.startswith('Inc') else 'T (°C)', col=1); fig.update_layout(height=440, legend=dict(orientation='h', y=-0.25)); st.plotly_chart(fig, width='stretch')
        st.dataframe(sel.rename(columns={'run': 'Prova', 'ric': 'Ricetta', 'V': 'V', 'cfg': 'Config.', 'sale': 'Sale (%)', 'T0_purea': 'T0 purea (°C)', 'T0_sfere': 'T0 sfere (°C)', 'n': 'Punti', 't_max': 't max (s)', 'dT_P1': 'ΔT P1', 'dT_P2': 'ΔT P2', 'dT_P3': 'ΔT P3'}).round(2), hide_index=True, width='stretch')
    ui.sezione("Qualità dei dati e rumore di misura")
    q1, q2 = st.columns(2)
    with q1:
        fig = go.Figure(go.Histogram(x=P.T0_purea, nbinsx=24, marker_color=ui.COL['teal'])); fig.add_vline(x=20, line=dict(color=ui.COL['rosso'], dash='dash'), annotation_text='20 °C (nominale del modello)')
        ui.layout(fig, 'Temperatura iniziale delle prove', 360, xaxis_title='T0 purea (°C)', yaxis_title='Prove'); st.plotly_chart(fig, width='stretch')
        st.caption(f"{int((P.T0_purea < 16).sum())} prove su {len(P)} partono sotto 16 °C: il confronto con le misure usa la temperatura iniziale di ciascuna prova e di ciascuna polpetta.")
    with q2:
        ud = calibra.uD_stima(1.5); Rp = calibra.ripetizioni(1.5); x = np.linspace(0, max(60, Rp.dT.max()), 100)
        fig = go.Figure(go.Scatter(x=Rp.dT, y=Rp.sd, mode='markers', marker=dict(size=5, color=ui.COL['strutt'], opacity=0.45), name='Ripetizioni')); fig.add_trace(go.Scatter(x=x, y=calibra.uD(x, ud), mode='lines', line=dict(color=ui.COL['caldo'], width=3), name='u_D(ΔT)'))
        ui.layout(fig, 'Modello del rumore di misura', 360, xaxis_title='Incremento medio ΔT (K)', yaxis_title='Deviazione standard fra ripetizioni (K)'); st.plotly_chart(fig, width='stretch')
        st.latex(r"u_D(\Delta T)=\sqrt{(a\,\Delta T)^2+b^2},\qquad a=%.3f,\; b=%.2f\ \mathrm{K}" % (ud['a'], ud['b']))
        st.caption("Stima dalle ripetizioni con stesse condizioni e temperatura iniziale entro 1.5 K: include la variabilità fra prove, non solo il rumore del sensore.")
ui.piede()
