"""Verifica numerica e incertezza: convergenza di mesh e di passo temporale, soglie, confronto fra le famiglie, verifica rapida in diretta."""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import ui, dati, analisi
from core import motore as mo

ui.applica_tema()
ui.hero("Verifica numerica e incertezza", "Convergenza di mesh e di passo temporale con GCI e minimi quadrati, tre esiti di convergenza e soglie per grandezza, confronto fra griglie strutturate e mesh conformi, e una verifica rapida eseguita in diretta.", ["75 casi × 13 grandezze", "5 livelli: 800 – 51 200 nodi", "GCI · minimi quadrati"])
D = dati.convergenza(); FN = {'strutturata': 'Griglie strutturate', 'conforme': 'Mesh conformi'}; CF = {'strutturata': ui.COL['strutt'], 'conforme': ui.COL['conf']}
tabs = st.tabs(["Convergenza di mesh", "Mappe e soglie", "Passo temporale", "Confronto fra le famiglie", "Verifica rapida in diretta"])

# ------------------------------------------------------------------------------------------------ 1
with tabs[0]:
    L = dati.livelli(); c = st.columns(5); stu = c[0].radio("Studio", ['A', 'B'], format_func=lambda s: dati.STUDI[s], key="vf_stu"); ric = c[1].selectbox("Ricetta", dati.RICS, index=4, format_func=lambda r: f"{r} · c = {dati.CONTR[r]:.2f}", key="vf_ric")
    V = c[2].selectbox("Tensione (V)", list(dati.VS), index=1, disabled=(stu == 'A'), key="vf_V"); cfg = c[3].selectbox("Configurazione", dati.CFG, key="vf_cfg")
    qs = sorted(D[D.Studio == stu].Grandezza.unique(), key=lambda q: list(dati.Q).index(q)); q = c[4].selectbox("Grandezza", qs, format_func=lambda q: f"{dati.Q[q][0]} · {dati.Q[q][1]}", key=f"vf_q_{stu}")
    Vu = 40 if stu == 'A' else V; off = 20.0 if q.startswith('T_') or q in ('T_max',) else 0.0
    fig = make_subplots(rows=1, cols=2, subplot_titles=[FN['strutturata'], FN['conforme']], horizontal_spacing=0.09); filas = []
    for j, fam in enumerate(['strutturata', 'conforme'], 1):
        s = L[(L.famiglia == fam) & (L.studio == stu) & (L.ric == ric) & (L.config == cfg) & (L.grandezza == q) & (L.V == Vu)].sort_values('h_rep_mm'); d = D[(D.Famiglia == fam) & (D.Studio == stu) & (D.Ricetta == ric) & (D.Tensione_V == Vu) & (D.Configurazione == cfg) & (D.Grandezza == q)]
        if s.empty or d.empty: continue
        d = d.iloc[0]; U = d.U_GCI_pc_scala / 100 * d.Scala; lg, ll = d.Limite_GCI - off, d.Limite_LS - off; x = s.h_rep_mm.values; y = s.valore.values - off
        fig.add_trace(go.Scatter(x=[x.min() * 0.85, x.max() * 1.1], y=[lg + U] * 2, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=j); fig.add_trace(go.Scatter(x=[x.min() * 0.85, x.max() * 1.1], y=[lg - U] * 2, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(250,176,176,0.45)', name='Incertezza U_h (GCI)', showlegend=(j == 1), hoverinfo='skip'), row=1, col=j)
        fig.add_trace(go.Scatter(x=[x.min() * 0.85, x.max() * 1.1], y=[lg] * 2, mode='lines', line=dict(color='#C92A2A', dash='dash', width=2), name='Limite GCI', showlegend=(j == 1)), row=1, col=j); fig.add_trace(go.Scatter(x=[x.min() * 0.85, x.max() * 1.1], y=[ll] * 2, mode='lines', line=dict(color='#2B8A3E', dash='dot', width=2.5), name='Limite minimi quadrati', showlegend=(j == 1)), row=1, col=j)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', line=dict(color=CF[fam], width=2.5), marker=dict(size=9), name='Calcolato', showlegend=(j == 1), customdata=s.n_nodi.values, hovertemplate='h=%{x:.2f} mm<br>%{customdata:,} nodi<br>valore=%{y:.4g}<extra></extra>'), row=1, col=j)
        filas.append({'Famiglia': FN[fam], 'Valore sulla mesh più fine': y[0], 'Limite GCI': lg, 'p (GCI)': d.p_GCI, 'Regime': d.Regime, 'U GCI (% scala)': d.U_GCI_pc_scala, 'Esito GCI': d.Esito_GCI, 'Limite LS': ll, 'p (LS)': d.p_LS, 'Tipo di fit': d.Ramo_LS, 'U LS (% scala)': d.U_LS_pc_scala, 'Esito LS': d.Esito_LS, 'Soglia (%)': d.Soglia_pc})
    fig.update_xaxes(type='log', title='h = (V/N)^(1/3) (mm)', tickvals=[2, 3, 4, 6, 8], ticktext=['2', '3', '4', '6', '8']); fig.update_yaxes(title=(dati.Q[q][1] + (' (incremento, K)' if off else f" ({dati.Q[q][2]})")), col=1); fig.update_layout(height=470, legend=dict(orientation='h', y=-0.2)); st.plotly_chart(fig, width='stretch')
    if filas: st.dataframe(pd.DataFrame(filas).round(4), hide_index=True, width='stretch')
    ui.callout("<b>Come leggere il grafico.</b> I marcatori sono i valori ai cinque livelli di mesh (a destra la più grossolana). La linea rossa è il limite del GCI sulle tre mesh più fini, la verde quello ai minimi quadrati su tutti i livelli, la banda rosa l'incertezza spaziale del GCI. Esito: <i>dimostrata</i> (regime monotono e incertezza ≤ soglia), <i>non dimostrata</i> (regime non monotono ma incertezza ≤ soglia con il ripiego conservativo) o <i>sopra soglia</i>.")

# ------------------------------------------------------------------------------------------------ 2
with tabs[1]:
    c = st.columns([1.2, 1, 1.4, 1]); modo = c[0].radio("Soglia", ["Per grandezza (2.5% / 5%)", "Uniforme"], key="vf_modo"); thr_u = c[1].slider("Soglia uniforme (%)", 0.5, 20.0, 2.5, 0.5, disabled=modo.startswith('Per'), key="vf_thr")
    stim = c[2].radio("Stimatore", ["GCI", "Minimi quadrati"], horizontal=True, key="vf_stim"); st_map = c[3].radio("Studio", ['A', 'B', 'C'], horizontal=True, key="vf_stm")
    cu = 'U_GCI_pc_scala' if stim == 'GCI' else 'U_LS_pc_scala'; E = D.copy(); E['thr'] = E.Soglia_pc if modo.startswith('Per') else thr_u; E['ok'] = E[cu] <= E.thr; E['dim'] = E.ok & (E.Regime == 'monotona')
    E['esito'] = np.where(~E.ok, 'Sopra soglia', np.where(E.dim, 'Dimostrata', 'Non dimostrata')); k = st.columns(4)
    for col, fam in zip(k[:2], ['strutturata', 'conforme']):
        d = E[(E.Famiglia == fam) & (E.Studio.isin(['A', 'B']))]; ui.kpi(col, f"{100 * d.ok.mean():.0f}%", f"{FN[fam]}: sotto soglia", f"convergenza dimostrata {100 * d.dim.mean():.0f}% · studi A e B")
    for col, fam in zip(k[2:], ['strutturata', 'conforme']):
        d = E[(E.Famiglia == fam) & (E.Studio == 'C')]; ui.kpi(col, f"{100 * d.ok.mean():.0f}%", f"{FN[fam]}: passo temporale", "casi sotto soglia")
    cols = st.columns(2)
    def mappa(fam, col):
        d = E[(E.Famiglia == fam) & (E.Studio == st_map)]; qs = [q for q in dati.Q if q in d.Grandezza.unique()]; righe = [(r, 40) for r in dati.RICS] if st_map == 'A' else [(r, v) for v in (50, 40, 30) for r in dati.RICS]
        M = np.full((len(righe), len(qs)), np.nan)
        for i, (r, v) in enumerate(righe):
            for j, q in enumerate(qs): x = d[(d.Ricetta == r) & (d.Tensione_V == v) & (d.Grandezza == q)]; M[i, j] = x.ok.sum() if len(x) else np.nan
        fig = go.Figure(go.Heatmap(z=M, x=[dati.Q[q][0] for q in qs], y=[(r if st_map == 'A' else f"{r} · {v} V") for r, v in righe], zmin=0, zmax=5, colorscale=[[0, '#b2182b'], [0.2, '#ef8a62'], [0.4, '#fddbc7'], [0.6, '#d9f0d3'], [0.8, '#7fbf7b'], [1, '#1b7837']], text=M, texttemplate='%{text:.0f}', colorbar=dict(title='su 5', tickvals=[0, 1, 2, 3, 4, 5])))
        fig.update_layout(height=520 if st_map != 'A' else 330, title=dict(text=FN[fam], x=0.01, font=dict(size=15)), yaxis=dict(autorange='reversed'), xaxis=dict(tickangle=-45)); col.plotly_chart(fig, width='stretch')
    mappa('strutturata', cols[0]); mappa('conforme', cols[1]); st.caption("Ogni cella: numero di configurazioni (su 5) con incertezza ≤ soglia. Nello studio A i risultati non dipendono dalla tensione.")
    ui.sezione("Esiti per grandezza"); d = E[E.Studio == st_map]; tab = d.groupby(['Famiglia', 'Grandezza']).esito.value_counts(normalize=True).unstack().fillna(0).mul(100).reset_index(); fig = go.Figure()
    for es, col in (('Dimostrata', '#2B8A3E'), ('Non dimostrata', '#E8B100'), ('Sopra soglia', '#C92A2A')):
        for fam in ['strutturata', 'conforme']:
            s = tab[tab.Famiglia == fam].set_index('Grandezza'); qs = [q for q in dati.Q if q in s.index]; fig.add_bar(x=[f"{dati.Q[q][0]} · {'S' if fam == 'strutturata' else 'C'}" for q in qs], y=[s.loc[q, es] if es in s.columns else 0 for q in qs], name=es, marker_color=col, legendgroup=es, showlegend=(fam == 'strutturata'))
    fig.update_layout(barmode='stack', height=420, yaxis_title='Casi (%)', xaxis=dict(tickangle=-60), legend=dict(orientation='h', y=1.08)); st.plotly_chart(fig, width='stretch'); st.caption("S: griglie strutturate; C: mesh conformi.")
    ui.sezione("Distribuzione cumulativa dell'incertezza"); sel = st.multiselect("Grandezze", [q for q in dati.Q if q in D[D.Studio.isin(['A', 'B'])].Grandezza.unique()], default=['T_P1', 'frac_P_p', 'q_media_polp'], format_func=lambda q: dati.Q[q][0], key="vf_ecdf")
    if sel:
        fig = make_subplots(rows=1, cols=len(sel), subplot_titles=[dati.Q[q][0] for q in sel], shared_yaxes=True, horizontal_spacing=0.04)
        for j, q in enumerate(sel, 1):
            for fam in ['strutturata', 'conforme']:
                u = np.sort(E[(E.Famiglia == fam) & (E.Grandezza == q) & E.Studio.isin(['A', 'B'])][cu].clip(lower=1e-3).values); y = 100 * np.arange(1, len(u) + 1) / len(u); fig.add_trace(go.Scatter(x=u, y=y, line=dict(color=CF[fam], width=2.6, shape='hv'), name=FN[fam], showlegend=(j == 1)), row=1, col=j)
            thq = E[E.Grandezza == q].thr.iloc[0]; fig.add_vline(x=thq, line=dict(color='#C92A2A', width=1.6), row=1, col=j)
        fig.update_xaxes(type='log', title='Incertezza (% scala)'); fig.update_yaxes(title='Casi con U ≤ x (%)', col=1); fig.update_layout(height=380, legend=dict(orientation='h', y=-0.25)); st.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------------------------------------------ 3
with tabs[2]:
    P = dati.passo_temporale(); c = st.columns(5); ric3 = c[0].selectbox("Ricetta", dati.RICS, index=4, key="vt_ric"); V3 = c[1].selectbox("Tensione (V)", list(dati.VS), index=1, key="vt_V"); cfg3 = c[2].selectbox("Configurazione", dati.CFG, key="vt_cfg"); q3 = c[3].selectbox("Grandezza", ['T_P1', 'T_P2', 'T_P3', 'T_max', 'E_in', 'P', 'frac_P_p'], format_func=lambda q: dati.Q[q][0], key="vt_q"); thr_t = c[4].slider("Soglia (%)", 0.5, 10.0, 2.5, 0.5, key="vt_thr")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Valore in funzione del passo", "Errore rispetto al limite per Δt → 0"), horizontal_spacing=0.10); off3 = 20.0 if q3.startswith('T_') else 0.0
    for fam in ['strutturata', 'conforme']:
        s = P[(P.famiglia == fam) & (P.ric == ric3) & (P.V == V3) & (P.config == cfg3) & (P.grandezza == q3)].sort_values('dt')
        if len(s) < 3: continue
        x = s.dt.values; y = s.valore.values - off3; est = analisi.stimatore(x, y + off3, 'ls'); f0 = est['f_best'] - off3; e = 100 * np.abs(y - f0) / max(abs(f0), 1e-12)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', line=dict(color=CF[fam], width=2.6), marker=dict(size=9), name=FN[fam]), row=1, col=1); fig.add_hline(y=f0, line=dict(color=CF[fam], dash='dot'), row=1, col=1); fig.add_trace(go.Scatter(x=x, y=np.maximum(e, 1e-4), mode='lines+markers', line=dict(color=CF[fam], width=2.6), marker=dict(size=9), showlegend=False), row=1, col=2)
    fig.add_hline(y=thr_t, line=dict(color='#C92A2A', dash='dot'), annotation_text='soglia', row=1, col=2); xr = np.array([5, 40]); fig.add_trace(go.Scatter(x=xr, y=0.8 * xr / 10, mode='lines', line=dict(color='black', width=1, dash='dash'), name='pendenza 1'), row=1, col=2)
    fig.update_xaxes(type='log', title='Δt (s)', tickvals=[5, 10, 20, 40], ticktext=['5', '10', '20', '40']); fig.update_yaxes(type='log', title='Errore (% del valore)', row=1, col=2); fig.update_yaxes(title=f"{dati.Q[q3][1]}", row=1, col=1); fig.update_layout(height=430, legend=dict(orientation='h', y=-0.22)); st.plotly_chart(fig, width='stretch')
    dt_t = analisi.dt_tabella(thr_t); k = st.columns(4); dd = dt_t[dt_t.grandezza == 'T_P1']
    ui.kpi(k[0], f"{dd.ordine.median():.2f}", "ordine osservato mediano (ΔT_P1)", "atteso 1 (Eulero implicito)"); ui.kpi(k[1], f"{dd.e10.median():.2f}%", "errore a 10 s, mediana", f"90° percentile {dd.e10.quantile(.9):.2f}%"); ui.kpi(k[2], f"{(dd.dt_richiesto >= 10).sum()}/{len(dd)}", f"casi con passo richiesto ≥ 10 s", f"soglia {thr_t:g}%"); ui.kpi(k[3], f"{dd.dt_richiesto.min():.1f} s", "passo richiesto minimo", "caso più restrittivo")
    a, b = st.columns(2)
    with a:
        gr = st.radio("Ordine osservato per", ['ric', 'V', 'config'], format_func=lambda x: {'ric': 'ricetta', 'V': 'tensione', 'config': 'configurazione'}[x], horizontal=True, key="vt_gr"); fig = go.Figure()
        for fam in ['strutturata', 'conforme']:
            x = dd[dd.famiglia == fam]; fig.add_trace(go.Box(x=x[gr].astype(str), y=x.ordine, name=FN[fam], marker_color=CF[fam], boxpoints=False))
        fig.add_hline(y=1, line=dict(color='black', dash='dash')); ui.layout(fig, "Ordine di convergenza nel tempo (ΔT_P1)", 380, boxmode='group', yaxis=dict(title='Ordine osservato', range=[0.3, 2.2]), legend=dict(orientation='h', y=-0.2)); st.plotly_chart(fig, width='stretch')
    with b:
        fam_h = st.radio("Famiglia", ['conforme', 'strutturata'], format_func=lambda f: FN[f], horizontal=True, key="vt_fh"); vh = st.select_slider("Tensione (V)", options=list(dati.VS), value=40, key="vt_vh"); x = dd[(dd.famiglia == fam_h) & (dd.V == vh)]; M = np.array([[x[(x.ric == r) & (x.config == c_)].dt_richiesto.iloc[0] for c_ in dati.CFG] for r in dati.RICS])
        fig = go.Figure(go.Heatmap(z=np.log10(np.minimum(M, 100)), x=dati.CFG, y=[f"{r} (c={dati.CONTR[r]:.2f})" for r in dati.RICS], zmid=1, zmin=np.log10(2), zmax=np.log10(100), colorscale='RdYlGn', text=np.where(M > 100, '>100', np.round(M, 0).astype(int).astype(str)), texttemplate='%{text}', colorbar=dict(title='Δt (s)', tickvals=np.log10([2, 5, 10, 20, 50, 100]), ticktext=['2', '5', '10', '20', '50', '100'])))
        ui.layout(fig, f"Passo massimo (s) con errore ≤ {thr_t:g}%", 380, yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
    st.caption("Il passo richiesto è estrapolato con l'ordine osservato; i valori oltre i passi provati vanno letti come «non limitante». La raccomandazione del protocollo considera tutte le grandezze e criteri più severi.")

# ------------------------------------------------------------------------------------------------ 4
with tabs[3]:
    EL = analisi.errore_livelli(); qx = st.selectbox("Grandezza", [q for q in dati.Q if q in EL.grandezza.unique()], format_func=lambda q: f"{dati.Q[q][0]} · {dati.Q[q][1]}", key="vc_q"); sub = EL[EL.grandezza == qx]; thq = float(sub.Soglia_pc.iloc[0]); fig = go.Figure(); testo = []
    for fam in ['strutturata', 'conforme']:
        g = sub[sub.famiglia == fam].groupby('livello'); n = g.n_nodi.median().values; med = g.err.median().values; lo = g.err.quantile(.25).values; hi = g.err.quantile(.75).values
        fig.add_trace(go.Scatter(x=np.r_[n, n[::-1]], y=np.r_[np.maximum(hi, 1e-4), np.maximum(lo, 1e-4)[::-1]], fill='toself', fillcolor='rgba(31,119,180,0.15)' if fam == 'strutturata' else 'rgba(255,127,14,0.15)', line=dict(width=0), hoverinfo='skip', showlegend=False)); fig.add_trace(go.Scatter(x=n, y=np.maximum(med, 1e-4), mode='lines+markers', line=dict(color=CF[fam], width=3), marker=dict(size=9), name=FN[fam]))
        if med[0] <= thq: nn = '≤ 800'
        elif med[-1] > thq: nn = '> 51 200'
        else: kk = int(np.where(med <= thq)[0][0]); nn = f"{np.exp(np.interp(np.log(thq), [np.log(med[kk]), np.log(med[kk - 1])], [np.log(n[kk]), np.log(n[kk - 1])])):,.0f}".replace(',', ' ')
        testo.append((fam, nn))
    fig.add_hline(y=thq, line=dict(color='#C92A2A', dash='dot'), annotation_text='soglia'); ui.layout(fig, "Errore in funzione dei nodi (mediana e quartili sui 75 casi)", 430, xaxis=dict(type='log', title='Numero di nodi'), yaxis=dict(type='log', title='Errore rispetto al riferimento comune (% della scala)'), legend=dict(orientation='h', y=-0.2)); a, b = st.columns([2.2, 1]); a.plotly_chart(fig, width='stretch')
    with b:
        st.markdown("**Nodi per raggiungere la soglia (mediana)**")
        for fam, nn in testo: ui.kpi(b, nn, FN[fam], f"soglia {thq:g}%"); b.write("")
    ui.callout("Il riferimento comune è il limite ai minimi quadrati della mesh conforme dello stesso caso: non è una soluzione indipendente. L'errore della conforme sui livelli fini è basso per costruzione; quello della strutturata comprende anche la differenza sistematica fra le due famiglie.", 'warn')
    a, b = st.columns(2)
    with a:
        G = D[D.Studio.isin(['A', 'B']) & (D.Grandezza == qx)].pivot_table(index=['Ricetta', 'Tensione_V', 'Configurazione'], columns='Famiglia', values='U_GCI_pc_scala').reset_index(); fig = go.Figure()
        for r, col in zip(dati.RICS, ui.COL_RIC): s = G[G.Ricetta == r]; fig.add_trace(go.Scatter(x=s.conforme.clip(lower=1e-3), y=s.strutturata.clip(lower=1e-3), mode='markers', marker=dict(color=col, size=8, line=dict(color='#212529', width=0.5)), name=f"{r} (c={dati.CONTR[r]:.2f})"))
        fig.add_trace(go.Scatter(x=[1e-3, 100], y=[1e-3, 100], mode='lines', line=dict(color='black', dash='dash'), name='parità')); fig.add_hline(y=thq, line=dict(color='#C92A2A', dash='dot')); fig.add_vline(x=thq, line=dict(color='#C92A2A', dash='dot'))
        ui.layout(fig, f"Incertezza caso per caso: conforme meglio nel {100 * (G.conforme < G.strutturata).mean():.0f}% dei casi", 420, xaxis=dict(type='log', title='U conforme (% scala)'), yaxis=dict(type='log', title='U strutturata (% scala)'), legend=dict(font=dict(size=11))); st.plotly_chart(fig, width='stretch')
    with b:
        I = dati.interfaccia(); sI = I[I.grandezza == qx]; fig = go.Figure()
        for r, col in zip(dati.RICS, ui.COL_RIC): fig.add_trace(go.Box(y=sI[sI.ric == r].E_rel_pc, name=f"{r}", marker_color=col, boxpoints='all', pointpos=0, jitter=0.35, marker_size=4))
        fig.add_hline(y=0, line=dict(color='black', width=1)); ui.layout(fig, f"Differenza fra i limiti delle famiglie, strutturata − conforme · oltre l'incertezza nel {100 * sI.oltre_incertezza.mean():.0f}% dei casi", 420, yaxis_title='% della scala', showlegend=False, xaxis_title='Ricetta (contrasto crescente da sinistra)'); st.plotly_chart(fig, width='stretch')
    R = dati.riepilogo(); R['t'] = R.tempo_s - R.G_durata.fillna(0); tm = R.groupby('famiglia').t.median()
    ui.sezione("Sintesi delle prestazioni"); rows = []
    for st_, nm in (('A', 'Studio A, elettrico'), ('B', 'Studio B, accoppiato'), ('C', 'Studio C, temporale')): rows.append({'Metrica': f'Casi con incertezza ≤ soglia (GCI), {nm} (%)', 'Griglie strutturate': round(100 * (D[(D.Famiglia == 'strutturata') & (D.Studio == st_)].Esito_GCI != 'Sopra soglia').mean()), 'Mesh conformi': round(100 * (D[(D.Famiglia == 'conforme') & (D.Studio == st_)].Esito_GCI != 'Sopra soglia').mean())})
    ab = D[D.Studio.isin(['A', 'B'])]; rows.append({'Metrica': 'Casi con convergenza monotona, studi A e B (%)', 'Griglie strutturate': round(100 * (ab[ab.Famiglia == 'strutturata'].Regime == 'monotona').mean()), 'Mesh conformi': round(100 * (ab[ab.Famiglia == 'conforme'].Regime == 'monotona').mean())})
    mo_ = ab[(ab.Regime == 'monotona') & np.isfinite(ab.p_GCI)]; rows.append({'Metrica': 'Ordine osservato mediano, casi monotoni', 'Griglie strutturate': round(mo_[mo_.Famiglia == 'strutturata'].p_GCI.median(), 2), 'Mesh conformi': round(mo_[mo_.Famiglia == 'conforme'].p_GCI.median(), 2)}); rows.append({'Metrica': 'Tempo di calcolo mediano per caso, escluso il globale (s)', 'Griglie strutturate': round(tm.get('strutturata', np.nan)), 'Mesh conformi': round(tm.get('conforme', np.nan))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch'); st.caption("I tempi dipendono dalla macchina e dal numero di core; sono indicativi del rapporto fra le famiglie.")

# ------------------------------------------------------------------------------------------------ 5
with tabs[4]:
    ui.callout("Raffinamento della mesh eseguito <b>in questa sessione</b> con il motore FEM, su un caso a scelta: utile per vedere come i valori si assestano. Con pochi livelli la stima del limite è meno affidabile di quella del protocollo (cinque livelli).")
    c = st.columns(6); r5 = c[0].selectbox("Ricetta", sorted(dati.AB), index=2, key="vr_ric"); V5 = c[1].selectbox("Tensione (V)", list(dati.VS), index=1, key="vr_V"); cfg5 = c[2].selectbox("Configurazione", dati.CFG, key="vr_cfg")
    fams = ['strutturata'] + (['conforme'] if mo.GMSH else []); fam5 = c[3].multiselect("Famiglie", fams, default=fams, format_func=lambda f: FN[f], key="vr_fam"); lmax = c[4].select_slider("Livello massimo", options=[1, 2, 3], value=2, format_func=lambda k: f"{k} ({mo.NODI_LIVELLO[k]:,} nodi)".replace(',', ' '), key="vr_lmax"); dur5 = c[5].number_input("Durata (s)", 30.0, 270.0, 120.0, 30.0, key="vr_dur")
    costo = sum(mo.stima_costo(cfg5, k) for k in range(lmax + 1)) * len(fam5) * dur5 / 270; st.caption(f"Tempo di calcolo stimato: circa {costo:.0f} s.")
    if st.button("▶ Esegui la verifica rapida", type="primary", disabled=not fam5, key="vr_go"):
        barra = st.progress(0.0, text="Raffinamento in corso…")
        try: st.session_state['vr'] = analisi.verifica_rapida(r5, V5, cfg5, fam5, list(range(lmax + 1)), 10.0, dur5, avanz=lambda x: barra.progress(min(1.0, x), text=f"Raffinamento in corso… {100 * x:.0f}%"))
        except Exception as e: st.error(f"Verifica non riuscita: {e}")
        barra.empty()
    VR = st.session_state.get('vr')
    if VR is not None and len(VR):
        qs5 = [('T_P1', 'ΔT alla sonda P1 (K)'), ('T_max', 'ΔT massimo (K)'), ('E_in', 'Energia fornita (J)'), ('frac_P_p', 'Π_p (t = 0)'), ('q_media_polp', 'q̄_p (t = 0, W/m³)')]; fig = make_subplots(rows=1, cols=len(qs5), subplot_titles=[b for a, b in qs5], horizontal_spacing=0.06); res = []
        for j, (q, lab) in enumerate(qs5, 1):
            for fam in VR.famiglia.unique():
                s = VR[VR.famiglia == fam].sort_values('h_mm'); fig.add_trace(go.Scatter(x=s.h_mm, y=s[q], mode='lines+markers', line=dict(color=CF[fam], width=2.5), marker=dict(size=9), name=FN[fam], showlegend=(j == 1)), row=1, col=j)
                if len(s) >= 3:
                    e = analisi.stimatore(s.h_mm.values[:3], s[q].values[:3], 'gci3'); fig.add_hline(y=e['f_best'], line=dict(color=CF[fam], dash='dot', width=1.5), row=1, col=j); res.append({'Famiglia': FN[fam], 'Grandezza': lab, 'Valore (mesh più fine)': s[q].values[0], 'Limite GCI': e['f_best'], 'Ordine': e['p'], 'Regime': e['regime'], 'U (assoluta)': e['U'], 'U (% del valore)': 100 * e['U'] / max(abs(s[q].values[0]), 1e-12)})
        fig.update_xaxes(type='log', title='h (mm)'); fig.update_layout(height=400, legend=dict(orientation='h', y=-0.3)); st.plotly_chart(fig, width='stretch')
        st.dataframe(VR.round(4).rename(columns={'h_mm': 'h (mm)', 'tempo_s': 'Tempo (s)', 'n_nodi': 'Nodi'}), hide_index=True, width='stretch')
        if res: st.markdown("**Stima del limite (GCI sui tre livelli più fini)**"); st.dataframe(pd.DataFrame(res).round(4), hide_index=True, width='stretch')
ui.piede()
