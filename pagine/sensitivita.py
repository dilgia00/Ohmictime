"""Sensitività locale e globale: risultati del protocollo per contrasto, tensione e configurazione, e calcolo locale in diretta."""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import ui, dati, calibra, analisi
from core import motore as mo

ui.applica_tema()
ui.hero("Sensitività dei parametri", "Quali parametri governano le temperature alle sonde, e come cambia l'importanza al variare del contrasto di conducibilità, della tensione e della configurazione delle polpette. Locale (effetti per +10%) e globale (indici di Sobol).", ["9 parametri", "locale · Morris · Sobol", "identificabilità"])
FN = {'strutturata': 'Griglie strutturate', 'conforme': 'Mesh conformi'}; ABBR = {('A', 50), ('B', 40), ('B', 50)}
GRUPPI = [('sig_m20', ['sig_m20'], 'σ₂₀,m'), ('sig_p20', ['sig_p20'], 'σ₂₀,p'), ('fc_m', ['fc_m'], 'f_c,m'), ('fc_p', ['fc_p'], 'f_c,p'), ('alfa', ['alfa_m', 'alfa_p'], 'α_m, α_p'), ('altri', ['fk_m', 'fk_p', 'fh'], 'f_k,m, f_k,p, f_h')]
GCOL = {'sig_m20': '#1f77b4', 'sig_p20': '#17becf', 'fc_m': '#d62728', 'fc_p': '#ff9896', 'alfa': '#2ca02c', 'altri': '#bdbdbd'}
tabs = st.tabs(["Sensitività locale", "Sensitività globale", "Calcolo locale in diretta"])

# ------------------------------------------------------------------------------------------------ locale
with tabs[0]:
    SL = dati.sens_locale(); c = st.columns(4); fam = c[0].radio("Famiglia", ['conforme', 'strutturata'], format_func=lambda f: FN[f], horizontal=True, key="sl_fam"); fase = c[1].radio("Sonde", ['tutte', 'polpetta', 'purea'], horizontal=True, key="sl_fase")
    mis = c[2].radio("Misura", ["Quota sul totale (%)", "Effetto di +10% (K)"], horizontal=True, key="sl_mis"); vista = c[3].selectbox("Vista", ["Mappa ricetta × parametro", "Quote per contrasto e tensione", "Per configurazione", "Dipendenza dalla tensione", "Fasi e contrasto"], key="sl_vista")
    d = SL[(SL.famiglia == fam) & (SL.sonde == fase)].copy(); d['tot'] = d.groupby('id').effetto_10pc_C.transform('sum'); d['quota'] = 100 * d.effetto_10pc_C / d.tot; val = 'quota' if mis.startswith('Quota') else 'effetto_10pc_C'; un = '%' if val == 'quota' else 'K'
    if fase == 'purea': st.caption("La configurazione Tris non ha sonde in purea: nelle viste per sonde in purea si usano le altre quattro configurazioni.")
    if vista.startswith("Mappa"):
        V = st.select_slider("Tensione (V)", options=list(dati.VS), value=40, key="sl_V"); m = d[d.V == V].groupby(['ric', 'parametro'])[val].mean().unstack()[dati.PAR].loc[dati.RICS]
        fig = go.Figure(go.Heatmap(z=m.values, x=[dati.PAR_TXT[p] for p in dati.PAR], y=[f"{r} · sale {dati.SALE[r]:g}% · c={dati.CONTR[r]:.2f}" for r in dati.RICS], colorscale='Viridis' if val != 'quota' else 'Blues', text=np.round(m.values, 2 if val != 'quota' else 0), texttemplate='%{text}', colorbar=dict(title=un)))
        ui.layout(fig, f"{mis} · {V} V · media sulle configurazioni", 420, yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
    elif vista.startswith("Quote"):
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{v} V" for v in dati.VS], shared_yaxes=True, horizontal_spacing=0.03)
        for j, V in enumerate(dati.VS, 1):
            for g, pl, lab in GRUPPI:
                y = [d[(d.V == V) & (d.ric == r) & d.parametro.isin(pl)].groupby('id')[val].sum().mean() for r in dati.RICS]; fig.add_trace(go.Bar(x=[f"{r}<br>c={dati.CONTR[r]:.2f}" for r in dati.RICS], y=y, name=lab, marker_color=GCOL[g], legendgroup=g, showlegend=(j == 1)), row=1, col=j)
        fig.update_layout(barmode='stack', height=460, legend=dict(orientation='h', y=-0.22)); fig.update_yaxes(title=un, col=1); st.plotly_chart(fig, width='stretch'); st.caption("Ricette in ordine di contrasto di conducibilità crescente. Il contrasto varia insieme al livello di sale della purea: l'effetto del contrasto non è isolato.")
    elif vista.startswith("Per conf"):
        V = st.select_slider("Tensione (V)", options=list(dati.VS), value=30, key="sl_Vc"); fig = make_subplots(rows=1, cols=5, subplot_titles=[f"{c_} ({dati.NPOLP[c_]})" for c_ in dati.CFG], shared_yaxes=True, horizontal_spacing=0.025)
        for j, c_ in enumerate(dati.CFG, 1):
            for g, pl, lab in GRUPPI:
                y = [d[(d.V == V) & (d.ric == r) & (d.config == c_) & d.parametro.isin(pl)][val].sum() for r in dati.RICS]; fig.add_trace(go.Bar(x=dati.RICS, y=y, name=lab, marker_color=GCOL[g], legendgroup=g, showlegend=(j == 1)), row=1, col=j)
        fig.update_layout(barmode='stack', height=460, legend=dict(orientation='h', y=-0.2)); fig.update_yaxes(title=un, col=1); st.plotly_chart(fig, width='stretch'); st.caption("Con tre polpette (Tris) tutte le sonde sono in polpetta e domina f_c,p; con una polpetta (Middle, Side) dominano σ₂₀,m e f_c,m. Le ricette sono in ordine di contrasto crescente (B, A, E, D, C).")
    elif vista.startswith("Dipend"):
        pars = st.multiselect("Parametri", dati.PAR, default=['sig_m20', 'sig_p20', 'fc_m', 'fc_p'], format_func=lambda p: dati.PAR_TXT[p], key="sl_pars"); fig = make_subplots(rows=1, cols=5, subplot_titles=[f"{r} (c={dati.CONTR[r]:.2f})" for r in dati.RICS], shared_yaxes=True, horizontal_spacing=0.03); esp = []
        for j, r in enumerate(dati.RICS, 1):
            for p in pars:
                y = np.array([d[(d.ric == r) & (d.V == V_) & (d.parametro == p)].effetto_10pc_C.mean() for V_ in dati.VS]); fig.add_trace(go.Scatter(x=list(dati.VS), y=y, mode='lines+markers', line=dict(color=ui.COL_PAR[p], width=2.6), marker=dict(size=9, symbol=['circle-open' if (r, V_) in ABBR else 'circle' for V_ in dati.VS]), name=dati.PAR_TXT[p], legendgroup=p, showlegend=(j == 1)), row=1, col=j)
                ok = [i for i, V_ in enumerate(dati.VS) if (r, V_) not in ABBR]
                if len(ok) >= 2 and np.all(y[ok] > 0): esp.append({'Ricetta': r, 'Parametro': dati.PAR_TXT[p], 'n': np.polyfit(np.log(np.array(dati.VS)[ok]), np.log(y[ok]), 1)[0]})
        fig.update_xaxes(type='log', tickvals=list(dati.VS), ticktext=[str(v) for v in dati.VS], title='V'); fig.update_yaxes(type='log', title='Effetto di +10% (K)', col=1); fig.update_layout(height=450, legend=dict(orientation='h', y=-0.25)); st.plotly_chart(fig, width='stretch')
        st.caption("Marcatore vuoto: tempo di valutazione abbreviato per il limite di 90 °C (A a 50 V, B a 40 e 50 V): non confrontabile a parità di tempo.")
        if esp: st.markdown("**Esponente n della tensione (effetto ∝ Vⁿ), dai casi a 270 s**"); st.dataframe(pd.DataFrame(esp).pivot(index='Ricetta', columns='Parametro', values='n').round(2), width='stretch')
    else:
        P2 = SL[(SL.famiglia == fam) & SL.sonde.isin(['polpetta', 'purea'])].copy(); P2['quota'] = 100 * P2.effetto_10pc_C / P2.groupby(['id', 'sonde']).effetto_10pc_C.transform('sum'); cr = [dati.CONTR[r] for r in dati.RICS]; fig = make_subplots(rows=1, cols=2, subplot_titles=("Sonde in polpetta", "Sonde in purea"), shared_yaxes=True, horizontal_spacing=0.05)
        for j, fs in enumerate(['polpetta', 'purea'], 1):
            for p in ['sig_m20', 'sig_p20', 'fc_m', 'fc_p', 'alfa_m']: y = [P2[(P2.sonde == fs) & (P2.ric == r) & (P2.parametro == p)].quota.mean() for r in dati.RICS]; fig.add_trace(go.Scatter(x=cr, y=y, mode='lines+markers', line=dict(color=ui.COL_PAR[p], width=2.8), marker=dict(size=9), name=dati.PAR_TXT[p], legendgroup=p, showlegend=(j == 1)), row=1, col=j)
        fig.update_xaxes(type='log', title='Contrasto σ_p/σ_m', tickvals=cr, ticktext=[f"{c_:.2f}" for c_ in cr]); fig.update_yaxes(title='Quota (%)', col=1); fig.update_layout(height=430, legend=dict(orientation='h', y=-0.25)); st.plotly_chart(fig, width='stretch')
        st.caption("Sulle sonde in polpetta, la quota di σ₂₀,m cresce con il contrasto e quella di σ₂₀,p cala: un'interpretazione plausibile (non verificata) è che, quando la polpetta conduce molto più della purea, la corrente sia limitata dalla purea in serie.")
    ui.sezione("Identificabilità (protocollo)"); I = SL[(SL.famiglia == fam) & (SL.sonde == 'tutte') & (SL.parametro == 'sig_m20')].copy(); I['cond'] = I.cond
    a, b = st.columns(2)
    with a:
        m = SL[(SL.famiglia == fam) & (SL.sonde == 'tutte')].groupby(['ric', 'parametro']).selezionato.mean().unstack()[dati.PAR].loc[dati.RICS] * 100
        fig = go.Figure(go.Heatmap(z=m.values, x=[dati.PAR_TXT[p] for p in dati.PAR], y=dati.RICS, colorscale='Greens', zmin=0, zmax=100, text=np.round(m.values, 0), texttemplate='%{text}', colorbar=dict(title='%'))); ui.layout(fig, "Parametro selezionato per la calibrazione (% dei 15 casi)", 360, yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
    with b:
        cm = I.groupby(['ric', 'V']).cond.median().unstack().loc[dati.RICS]; fig = go.Figure(go.Heatmap(z=np.log10(cm.values), x=[f"{v} V" for v in cm.columns], y=dati.RICS, colorscale='YlOrRd', text=[[f"{x:.1e}" for x in r] for r in cm.values], texttemplate='%{text}', colorbar=dict(title='log10'))); ui.layout(fig, "Numero di condizionamento della matrice di sensitività", 360, yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
    st.caption("Il numero di condizionamento è dell'ordine di 10⁵–10⁶: i parametri non sono tutti identificabili insieme dalle sole temperature; il protocollo ne seleziona un sottoinsieme (di norma σ₂₀,m, f_c,m, f_c,p).")

# ------------------------------------------------------------------------------------------------ globale
with tabs[1]:
    So = dati.sobol(); Mo = dati.morris(); Q2 = dati.qualita(); USC = {'rms_sonde': 'Valore efficace degli incrementi alle sonde', 'T_media_sonde_finale': 'Temperatura media finale alle sonde', 'incremento_T_max_finale': 'Incremento finale della temperatura massima'}
    c = st.columns(4); us = c[0].selectbox("Uscita", list(USC), format_func=lambda u: USC[u], key="sg_us"); idx = c[1].radio("Indice", ["S_T (totale)", "S_1 (primo ordine)", "μ* (Morris, quota %)"], key="sg_idx"); vista = c[2].selectbox("Vista", ["Mappa ricetta × parametro", "Per contrasto e tensione", "Per configurazione", "Parametro dominante", "Dipendenza dalla tensione", "Accordo con la locale", "Intervalli e qualità"], key="sg_vista")
    if idx.startswith('S_T'): d = So[So.uscita == us].rename(columns={'ST': 'val'})
    elif idx.startswith('S_1'): d = So[So.uscita == us].rename(columns={'S1': 'val'})
    else: d = Mo[Mo.uscita == us].rename(columns={'quota_mu_star_pc': 'val'}); d['val'] = d['val'] / 100
    if vista.startswith("Mappa"):
        cV = c[3].select_slider("Tensione (V)", options=list(dati.VS), value=40, key="sg_V"); cc = st.selectbox("Configurazione", ['media sulle 5'] + dati.CFG, key="sg_cfg"); x = d[d.V == cV]
        if cc != 'media sulle 5': x = x[x.cfg == cc]
        m = x.groupby(['ric', 'parametro']).val.mean().unstack()[dati.PAR].loc[dati.RICS]; fig = go.Figure(go.Heatmap(z=m.values, x=[dati.PAR_TXT[p] for p in dati.PAR], y=[f"{r} · c={dati.CONTR[r]:.2f}" for r in dati.RICS], colorscale='Viridis', zmin=0, zmax=0.7, text=np.round(m.values, 2), texttemplate='%{text}', colorbar=dict(title=idx.split()[0])))
        ui.layout(fig, f"{idx} · {cV} V · {cc}", 420, yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
        a, b = st.columns(2); a.plotly_chart(go.Figure([go.Box(y=d[d.parametro == p].val, name=dati.PAR_TXT[p], marker_color=ui.COL_PAR[p], boxpoints=False) for p in dati.PAR]).update_layout(height=340, title=dict(text='Distribuzione sui 75 casi', x=0.01, font=dict(size=15)), showlegend=False, yaxis_title=idx.split()[0]), width='stretch')
        iv = So[So.uscita == us].groupby('caso').apply(lambda g: g.ST.sum() - g.S1.sum(), include_groups=False); b.metric("Somma ST − somma S1 (interazioni), massimo sui casi", f"{iv.max():.3f}"); b.caption("Valori vicini a zero indicano una risposta praticamente additiva nel metamodello usato.")
    elif vista.startswith("Per contrasto"):
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{v} V" for v in dati.VS], shared_yaxes=True, horizontal_spacing=0.03)
        for j, V in enumerate(dati.VS, 1):
            for g, pl, lab in GRUPPI: y = [d[(d.V == V) & (d.ric == r) & d.parametro.isin(pl)].groupby('caso').val.sum().mean() for r in dati.RICS]; fig.add_trace(go.Bar(x=[f"{r}<br>c={dati.CONTR[r]:.2f}" for r in dati.RICS], y=y, name=lab, marker_color=GCOL[g], legendgroup=g, showlegend=(j == 1)), row=1, col=j)
        fig.update_layout(barmode='stack', height=470, legend=dict(orientation='h', y=-0.22)); fig.update_yaxes(title=idx.split()[0], col=1); st.plotly_chart(fig, width='stretch')
    elif vista.startswith("Per conf"):
        fig = make_subplots(rows=1, cols=5, subplot_titles=[f"{c_} ({dati.NPOLP[c_]})" for c_ in dati.CFG], shared_yaxes=True, horizontal_spacing=0.025)
        for j, c_ in enumerate(dati.CFG, 1):
            for g, pl, lab in GRUPPI: y = [d[(d.cfg == c_) & (d.ric == r) & d.parametro.isin(pl)].groupby('caso').val.sum().mean() for r in dati.RICS]; fig.add_trace(go.Bar(x=dati.RICS, y=y, name=lab, marker_color=GCOL[g], legendgroup=g, showlegend=(j == 1)), row=1, col=j)
        fig.update_layout(barmode='stack', height=470, legend=dict(orientation='h', y=-0.2)); fig.update_yaxes(title=idx.split()[0], col=1); st.plotly_chart(fig, width='stretch'); st.caption("Media sulle tre tensioni; ricette in ordine di contrasto crescente (B, A, E, D, C).")
    elif vista.startswith("Parametro dom"):
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{v} V" for v in dati.VS], horizontal_spacing=0.04); pidx = {p: i for i, p in enumerate(dati.PAR)}
        for j, V in enumerate(dati.VS, 1):
            Z = np.zeros((5, 5)); T = [['' for _ in range(5)] for _ in range(5)]
            for i, r in enumerate(dati.RICS):
                for k, cfg in enumerate(dati.CFG):
                    s = d[(d.ric == r) & (d.V == V) & (d.cfg == cfg)].sort_values('val', ascending=False); Z[i, k] = pidx[s.parametro.iloc[0]]; T[i][k] = f"{dati.PAR_TXT[s.parametro.iloc[0]]}<br>{s.val.iloc[0]:.2f}"
            fig.add_trace(go.Heatmap(z=Z, x=dati.CFG, y=[f"{r} (c={dati.CONTR[r]:.2f})" for r in dati.RICS], zmin=0, zmax=8, text=T, texttemplate='%{text}', colorscale=[[i / 8, ui.COL_PAR[p]] for i, p in enumerate(dati.PAR)], showscale=False, xgap=2, ygap=2), row=1, col=j)
        fig.update_yaxes(autorange='reversed'); fig.update_layout(height=430); st.plotly_chart(fig, width='stretch'); st.caption("Colore e testo: parametro con l'indice maggiore e suo valore.")
    elif vista.startswith("Dipend"):
        pars = st.multiselect("Parametri", dati.PAR, default=['sig_m20', 'alfa_m', 'sig_p20', 'fc_m', 'fc_p'], format_func=lambda p: dati.PAR_TXT[p], key="sg_pars"); fig = make_subplots(rows=1, cols=5, subplot_titles=[f"{r} (c={dati.CONTR[r]:.2f})" for r in dati.RICS], shared_yaxes=True, horizontal_spacing=0.03)
        for j, r in enumerate(dati.RICS, 1):
            for p in pars: y = np.array([d[(d.ric == r) & (d.V == V_) & (d.parametro == p)].val.mean() for V_ in dati.VS]); fig.add_trace(go.Scatter(x=list(dati.VS), y=y, mode='lines+markers', line=dict(color=ui.COL_PAR[p], width=2.6), marker=dict(size=9, symbol=['circle-open' if (r, V_) in ABBR else 'circle' for V_ in dati.VS]), name=dati.PAR_TXT[p], legendgroup=p, showlegend=(j == 1)), row=1, col=j)
        fig.update_xaxes(tickvals=list(dati.VS), title='V'); fig.update_yaxes(title=idx.split()[0], col=1); fig.update_layout(height=450, legend=dict(orientation='h', y=-0.25)); st.plotly_chart(fig, width='stretch'); st.caption("Marcatore vuoto: tempo di valutazione abbreviato (limite di 90 °C). Nelle ricette valutate sempre a 270 s, il peso di α_m cresce con la tensione.")
    elif vista.startswith("Accordo"):
        LG = dati.loc_glob(); SLc = dati.sens_locale(); SLc = SLc[(SLc.famiglia == 'conforme') & (SLc.sonde == 'tutte')].copy(); fac = {'sig_m20': 2.0, 'alfa_m': 8.2, 'sig_p20': 2.0, 'alfa_p': 8.2, 'fk_m': 2.4, 'fk_p': 2.4, 'fc_m': 1.6, 'fc_p': 1.6, 'fh': 7.0}
        SLc['e'] = SLc.effetto_10pc_C * SLc.parametro.map(fac); SLc['q'] = SLc.e ** 2 / SLc.groupby('id').e.transform(lambda s: (s ** 2).sum()); DD = SLc.rename(columns={'id': 'caso'})[['caso', 'parametro', 'q']].merge(So[So.uscita == 'rms_sonde'][['caso', 'parametro', 'ST', 'cfg', 'ric']], on=['caso', 'parametro'])
        a, b = st.columns(2); fig = go.Figure()
        for c_ in dati.CFG: s = DD[DD.cfg == c_]; fig.add_trace(go.Scatter(x=s.q, y=s.ST, mode='markers', marker=dict(size=6, opacity=0.7), name=c_))
        fig.add_trace(go.Scatter(x=[0, 0.7], y=[0, 0.7], mode='lines', line=dict(color='black', dash='dash'), name='identità')); ui.layout(fig, f"Quota attesa dalla locale e S_T globale: r = {DD.q.corr(DD.ST):.3f} ({len(DD)} coppie)", 420, xaxis_title='Quota di varianza attesa dall\'analisi locale', yaxis_title='S_T globale'); a.plotly_chart(fig, width='stretch')
        fig = go.Figure()
        for r, col in zip(dati.RICS, ui.COL_RIC): fig.add_trace(go.Box(y=LG[LG.ric == r].rho_locale_sobolST, name=f"{r} (c={dati.CONTR[r]:.2f})", marker_color=col, boxpoints='all', pointpos=0, jitter=0.3, marker_size=4))
        ui.layout(fig, f"Correlazione di rango locale–globale (mediana {LG.rho_locale_sobolST.median():.2f}, minimo {LG.rho_locale_sobolST.min():.2f})", 420, yaxis_title='ρ di Spearman', showlegend=False); b.plotly_chart(fig, width='stretch')
    else:
        st.markdown("**Intervalli di incertezza assunti per i parametri**"); st.dataframe(dati.intervalli().rename(columns={'parametri': 'Parametri', 'semiampiezza_relativa': 'Semiampiezza relativa', 'scarto_tipo_di_letteratura': 'Scarto tipo da letteratura', 'fonte_e_nota': 'Fonte e nota', 'affidabilita_e_limite': 'Affidabilità e limite'}), hide_index=True, width='stretch')
        qq = Q2.groupby('uscita').agg(Q2_min=('Q2', 'min'), Q2_mediano=('Q2', 'median'), R2_min=('R2', 'min')).round(4); st.markdown("**Qualità del metamodello (polinomi del caos)**"); st.dataframe(qq, width='stretch'); ui.callout("Gli indici dipendono dagli intervalli assunti, da letteratura e non da misure: il peso di α_m, per esempio, riflette l'ampiezza (±41%) del suo intervallo.", 'warn')

# ------------------------------------------------------------------------------------------------ in diretta
with tabs[2]:
    ui.callout("Sensitività locale calcolata <b>in questa sessione</b> con il motore FEM: ogni parametro è aumentato del 10% e si osserva la risposta delle tre temperature alle sonde. Servono 10 simulazioni (una di base e nove perturbate).")
    c = st.columns(6); r6 = c[0].selectbox("Ricetta", sorted(dati.AB), index=2, key="sd_ric"); V6 = c[1].selectbox("Tensione (V)", list(dati.VS), index=1, key="sd_V"); cfg6 = c[2].selectbox("Configurazione", dati.CFG, key="sd_cfg")
    fams = ['strutturata'] + (['conforme'] if mo.GMSH else []); fam6 = c[3].selectbox("Famiglia", fams, index=len(fams) - 1, format_func=lambda f: FN[f], key="sd_fam"); lv6 = c[4].select_slider("Livello", options=[0, 1, 2], value=0, format_func=lambda k: f"{k} · {mo.NODI_LIVELLO[k]:,} nodi".replace(',', ' '), key="sd_lv"); dur6 = c[5].number_input("Durata (s)", 60.0, 270.0, 150.0, 30.0, key="sd_dur")
    sig = st.slider("Rumore di misura assunto σ (K)", 0.1, 1.0, 0.5, 0.05, key="sd_sig"); se_max = st.slider("Errore standard relativo massimo ammesso", 0.05, 0.5, 0.20, 0.05, key="sd_semax")
    st.caption(f"Tempo di calcolo stimato: circa {10 * mo.stima_costo(cfg6, lv6) * dur6 / 270:.0f} s.")
    if st.button("▶ Calcola la sensitività", type="primary", key="sd_go"):
        barra = st.progress(0.0, text="Calcolo in corso…")
        try: X, tempi, S0 = analisi.sensibilita_diretta(r6, V6, cfg6, fam6, lv6, 10.0, dur6, avanz=lambda x: barra.progress(min(1.0, x), text=f"Calcolo in corso… {100 * x:.0f}%")); st.session_state['sd'] = dict(X=X, tempi=tempi, S0=S0, caso=(r6, V6, cfg6, fam6, lv6))
        except Exception as e: st.error(f"Calcolo non riuscito: {e}")
        barra.empty()
    SD = st.session_state.get('sd')
    if SD:
        X = SD['X']; r_, V_, cfg_, fam_, lv_ = SD['caso']; sf = [i for i in range(3) if dati.OCC[cfg_][i]]; fasi = np.repeat(['polpetta' if i in sf else 'purea' for i in range(3)], len(SD['tempi'])); nrm = np.sqrt((X ** 2).mean(axis=0)); eff = 0.1 * nrm
        a, b = st.columns([1.1, 1])
        with a:
            fig = go.Figure(); o = np.argsort(eff); fig.add_bar(y=[dati.PAR_TXT[dati.PAR[i]] for i in o], x=eff[o], orientation='h', marker_color=[ui.COL_PAR[dati.PAR[i]] for i in o], name='In diretta')
            ref = dati.sens_locale(); ref = ref[(ref.id == f"{r_}_{V_}V__{cfg_}") & (ref.sonde == 'tutte') & (ref.famiglia == ('conforme' if fam_ == 'conforme' else 'strutturata'))].set_index('parametro').effetto_10pc_C
            if len(ref): fig.add_trace(go.Scatter(y=[dati.PAR_TXT[dati.PAR[i]] for i in o], x=[ref.get(dati.PAR[i], np.nan) for i in o], mode='markers', marker=dict(symbol='diamond', size=11, color='black'), name='Protocollo (5 livelli)'))
            ui.layout(fig, "Effetto di +10% sulle temperature alle sonde (K, valore efficace)", 430, xaxis_title='K', legend=dict(orientation='h', y=-0.15)); st.plotly_chart(fig, width='stretch')
        with b:
            Xn = X / np.linalg.norm(X, axis=0, keepdims=True).clip(min=1e-12); C = np.abs(Xn.T @ Xn); fig = go.Figure(go.Heatmap(z=C, x=[dati.PAR_TXT[p] for p in dati.PAR], y=[dati.PAR_TXT[p] for p in dati.PAR], zmin=0, zmax=1, colorscale='RdBu_r', zmid=0.5, text=np.round(C, 2), texttemplate='%{text}', textfont=dict(size=9))); ui.layout(fig, "Correlazione fra i parametri (|coseno| delle colonne)", 430, yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
        sv = np.linalg.svd(X, compute_uv=False); k = st.columns(3); ui.kpi(k[0], f"{sv[0] / max(sv[-1], 1e-300):.1e}", "numero di condizionamento", "tutti i parametri"); ui.kpi(k[1], f"{int((sv / sv[0] > 1e-3).sum())}", "direzioni indipendenti", "valori singolari > 0.1% del massimo")
        sel = []
        for j in np.argsort(-nrm):
            if nrm[j] <= 0: continue
            try: se_t = sig * np.sqrt(np.abs(np.diag(np.linalg.inv(X[:, sel + [j]].T @ X[:, sel + [j]]))))
            except np.linalg.LinAlgError: continue
            if np.all(se_t <= se_max): sel = sel + [j]
        ui.kpi(k[2], ", ".join(dati.PAR_TXT[dati.PAR[j]] for j in sel) or "nessuno", "parametri selezionabili", f"errore standard ≤ {se_max:g}, σ = {sig:g} K")
        fig = go.Figure(go.Bar(x=[dati.PAR_TXT[p] for p in dati.PAR], y=[sig / max(n, 1e-12) for n in nrm], marker_color=[ui.COL_PAR[p] for p in dati.PAR])); fig.add_hline(y=se_max, line=dict(color='#C92A2A', dash='dash'), annotation_text='limite')
        ui.layout(fig, "Errore standard relativo di ciascun parametro stimato da solo", 340, yaxis=dict(type='log', title='σ / norma'), xaxis_title=''); st.plotly_chart(fig, width='stretch')
        fig = go.Figure()
        for i, j in enumerate(range(3)): fig.add_trace(go.Scatter(x=SD['tempi'], y=SD['S0'][i], mode='lines', name=f"P{i + 1} ({'polpetta' if i in sf else 'purea'})", line=dict(width=3)))
        ui.layout(fig, "Incrementi di temperatura del caso di base", 330, xaxis_title='Tempo (s)', yaxis_title='ΔT (K)'); st.plotly_chart(fig, width='stretch')
ui.piede()
