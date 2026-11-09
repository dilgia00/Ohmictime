"""Dossier del caso: tutti i risultati del protocollo per una combinazione di ricetta, tensione e configurazione."""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import ui, dati, scenari as sc
from core import motore as mo
import calibrazione_protocollo as cp

ui.applica_tema()
ui.hero("Dossier del caso", "Per una combinazione di ricetta, tensione e configurazione delle polpette: conducibilità, riscaldamento previsto, esiti di convergenza e incertezza numerica delle due famiglie, passo temporale, sensitività locale e globale, mesh suggerite e confronto con le misure.", ["75 casi", "13 grandezze", "due famiglie di mesh", "sensitività locale e globale"])
c = st.columns(4); r = c[0].selectbox("Ricetta", dati.RICS, index=2, format_func=lambda x: f"{x} · sale {dati.SALE[x]:g}% · contrasto {dati.CONTR[x]:.2f}", key="do_r"); V = c[1].select_slider("Tensione (V)", list(dati.VS), value=40, key="do_v"); cf = c[2].selectbox("Configurazione", dati.CFG, format_func=lambda x: f"{x} · {dati.NPOLP[x]} polpett{'e' if dati.NPOLP[x] > 1 else 'a'}", key="do_c"); cid = f"{r}_{V}V__{cf}"
D = dati.convergenza(); D = D[(D.Ricetta == r) & (D.Tensione_V == V) & (D.Configurazione == cf)]; RI = dati.riepilogo(); RI = RI[RI.id == cid]; ES = {'Dimostrata': '🟢 dimostrata', 'Non dimostrata': '🟡 non dimostrata', 'Sopra soglia': '🔴 sopra soglia'}
rc = RI[RI.famiglia == 'conforme'].iloc[0] if len(RI[RI.famiglia == 'conforme']) else None; k = st.columns(5)
ui.kpi(k[0], f"{dati.sigma_T(r, 20):.2f} S/m", "σ della purea a 20 °C", f"polpette {dati.sigma_polp(20):.2f} S/m"); ui.kpi(k[1], f"{dati.CONTR[r]:.2f}", "contrasto di conducibilità", "σ polpette / σ purea a 20 °C")
if rc is not None: ui.kpi(k[2], f"{rc.dT_max_tfix:.1f} K", "riscaldamento massimo", f"a {rc.t_valutazione:.0f} s" + (" (tempo abbreviato)" if rc.t_valutazione < 270 else "")); ui.kpi(k[3], f"{rc.h_operativa_mm:.1f} mm" if np.isfinite(rc.h_operativa_mm) else "—", "mesh operativa suggerita (conforme)", "dimensione di riferimento h"); ui.kpi(k[4], f"{rc.C_dt_min:.0f} s" if np.isfinite(rc.C_dt_min) else "—", "passo temporale suggerito", "dal protocollo")
tab = st.tabs(["Esiti di convergenza", "Convergenza di mesh", "Passo temporale", "Sensitività", "Mesh e passi suggeriti", "Misure e modello nominale", "Scheda"])

# ------------------------------------------------------------------------------------------------ esiti
with tab[0]:
    ui.sezione("Incertezza numerica e esito per grandezza"); rows = []
    for q in dati.Q:
        row = {'Grandezza': dati.Q[q][0], 'Descrizione': dati.Q[q][1], 'Soglia (%)': dati.Q[q][3]}
        for fam, nome in (('strutturata', 'Strutt.'), ('conforme', 'Conf.')):
            x = D[(D.Famiglia == fam) & (D.Grandezza == q)]
            if len(x): x = x.iloc[0]; row[f'{nome} U (%)'] = x.U_GCI_pc_scala; row[f'{nome} esito'] = ES[x.Esito_GCI]; row[f'{nome} ordine p'] = x.p_GCI; row[f'{nome} regime'] = x.Regime
        rows.append(row)
    T = pd.DataFrame(rows); k = st.columns(4)
    for j, (fam, nome) in enumerate((('strutturata', 'griglie strutturate'), ('conforme', 'mesh conformi'))):
        x = D[D.Famiglia == fam]; n = int((x.Esito_GCI != 'Sopra soglia').sum()); d_ = int((x.Esito_GCI == 'Dimostrata').sum()); ui.kpi(k[2 * j], f"{n} su {len(x)}", f"grandezze ≤ soglia, {nome}", "numero di combinazioni caso-grandezza"); ui.kpi(k[2 * j + 1], f"{d_} su {len(x)}", f"convergenza dimostrata, {nome}", "regime monotono e ≤ soglia")
    st.dataframe(T.round(2), hide_index=True, width='stretch', height=560); ui.callout("L'esito «sopra soglia» non indica un errore del modello: segnala che la discretizzazione potrebbe alterare il confronto con le misure più di quanto accettato. Le soglie (2.5% per le grandezze legate alle misure, 5% per gli indicatori locali) sono quelle della tesi.")

# ------------------------------------------------------------------------------------------------ convergenza
with tab[1]:
    L = dati.livelli(); L = L[L.id == cid]; qs = st.multiselect("Grandezze", [q for q in dati.Q if q in L.grandezza.unique()], default=[q for q in ('T_P1', 'frac_P_p', 'q_media_polp', 'E_in') if q in L.grandezza.unique()], format_func=lambda q: f"{dati.Q[q][0]} · {dati.Q[q][1]}", key="do_q")
    if qs:
        nc = 2; nr = int(np.ceil(len(qs) / nc)); fig = make_subplots(rows=nr, cols=nc, subplot_titles=[dati.Q[q][0] for q in qs], horizontal_spacing=0.09, vertical_spacing=0.16 if nr > 1 else 0.1)
        for i, q in enumerate(qs):
            for fam, col in (('strutturata', ui.COL['strutt']), ('conforme', ui.COL['conf'])):
                x = L[(L.famiglia == fam) & (L.grandezza == q)].sort_values('h_rep_mm'); y = x.valore.values - (20.0 if q.startswith('T_') else 0.0)
                fig.add_trace(go.Scatter(x=x.h_rep_mm, y=y, mode='lines+markers', line=dict(color=col, width=2.5), marker=dict(size=8), name='strutturata' if fam == 'strutturata' else 'conforme', legendgroup=fam, showlegend=(i == 0)), row=i // nc + 1, col=i % nc + 1)
                dd = D[(D.Famiglia == fam) & (D.Grandezza == q)]
                if len(dd): off = 20.0 if q.startswith('T_') else 0.0; fig.add_hline(y=float(dd.iloc[0].Limite_LS) - off, line=dict(color=col, dash='dot', width=1.5), row=i // nc + 1, col=i % nc + 1)
        fig.update_xaxes(type='log', title='h (mm)', tickvals=[2, 3, 4, 6]); fig.update_layout(height=330 * nr, legend=dict(orientation='h', y=-0.1)); st.plotly_chart(fig, width='stretch'); st.caption("Valore ai cinque livelli di mesh (le temperature come incremento). Linee punteggiate: limite per h → 0 dai minimi quadrati. Le due famiglie hanno lo stesso numero di nodi a ogni livello.")

# ------------------------------------------------------------------------------------------------ passo temporale
with tab[2]:
    PT = dati.passo_temporale(); PT = PT[PT.id == cid]; a, b = st.columns(2)
    for ax, q, tit in ((a, 'T_P1', 'Incremento di T_P1 (K)'), (b, 'E_in', 'Energia fornita (J)')):
        fig = go.Figure()
        for fam, col in (('strutturata', ui.COL['strutt']), ('conforme', ui.COL['conf'])):
            x = PT[(PT.famiglia == fam) & (PT.grandezza == q)].sort_values('dt')
            if len(x): fig.add_trace(go.Scatter(x=x.dt, y=x.valore - (20.0 if q == 'T_P1' else 0.0), mode='lines+markers', line=dict(color=col, width=2.5), marker=dict(size=9), name=fam))
        ui.layout(fig, tit, 360, xaxis=dict(type='log', title='Passo temporale (s)', tickvals=[2.5, 5, 10, 20, 40]), legend=dict(orientation='h', y=-0.25)); ax.plotly_chart(fig, width='stretch')
    st.caption("L'errore del passo temporale decresce linearmente con Δt (ordine circa 0.9 per le temperature e per l'energia): il passo di 10 s ha errori mediani inferiori all'1% dell'incremento.")

# ------------------------------------------------------------------------------------------------ sensitività
with tab[3]:
    SL = dati.sens_locale(); SL = SL[(SL.id == cid) & (SL.famiglia == 'conforme')]; SO = dati.sobol(); SO = SO[(SO.caso == cid) & (SO.uscita == 'rms_sonde')]; a, b = st.columns(2); fig = go.Figure()
    for son, col in (('tutte', ui.COL['blu_scuro']), ('polpetta', ui.COL['conf']), ('purea', ui.COL['strutt'])):
        x = SL[SL.sonde == son].set_index('parametro').reindex(dati.PAR)
        if x.effetto_10pc_C.notna().any(): fig.add_bar(x=[dati.PAR_TXT[p] for p in dati.PAR], y=x.effetto_10pc_C, name={'tutte': 'tutte le sonde', 'polpetta': 'sonde in polpetta', 'purea': 'sonde in purea'}[son], marker_color=col)
    ui.layout(fig, "Sensitività locale: effetto di +10% del parametro (K)", 380, barmode='group', yaxis_title='K', legend=dict(orientation='h', y=-0.25)); a.plotly_chart(fig, width='stretch')
    x = SO.set_index('parametro').reindex(dati.PAR); fig = go.Figure(); fig.add_bar(x=[dati.PAR_TXT[p] for p in dati.PAR], y=x.ST, name='S_T (totale)', marker_color=ui.COL['caldo'], error_y=dict(type='data', symmetric=False, array=(x.ST_sup - x.ST).fillna(0), arrayminus=(x.ST - x.ST_inf).fillna(0))); fig.add_bar(x=[dati.PAR_TXT[p] for p in dati.PAR], y=x.S1, name='S_1 (primo ordine)', marker_color=ui.COL['teal'])
    ui.layout(fig, "Sensitività globale (Sobol, mesh conformi)", 380, barmode='group', yaxis_title='indice', legend=dict(orientation='h', y=-0.25)); b.plotly_chart(fig, width='stretch'); ident = SL[(SL.sonde == 'tutte')]
    if len(ident):
        sel = ident[ident.selezionato == True].parametro.tolist(); cond = float(ident.cond.iloc[0]); st.markdown(f"**Identificabilità.** Numero di condizionamento della matrice di sensitività: {cond:.1e}. Parametri selezionati per la calibrazione: " + (' '.join(ui.badge(dati.PAR_TXT[p], 'ok') for p in sel) if sel else '—'), unsafe_allow_html=True)
    ui.callout("Con tre polpette (Tris) domina f_c,p; con una (Middle, Side) domina σ₂₀,m; la tensione cambia soprattutto l'ampiezza degli effetti, non le quote.")

# ------------------------------------------------------------------------------------------------ mesh suggerite
with tab[4]:
    cols = ['famiglia', 't_valutazione', 'dT_max_tfix', 'h_raccomandata_mm', 'h_temperature_mm', 'h_operativa_mm', 'incertezza_operativa_pc', 'C_dt_min', 'nodi_predetti']; X = RI[[c_ for c_ in cols if c_ in RI.columns]].rename(columns={'famiglia': 'Famiglia', 't_valutazione': 'Tempo di valutazione (s)', 'dT_max_tfix': 'Riscaldamento massimo (K)', 'h_raccomandata_mm': 'h suggerita, soglia (mm)', 'h_temperature_mm': 'h suggerita, temperature (mm)', 'h_operativa_mm': 'h operativa (mm)', 'incertezza_operativa_pc': 'Incertezza operativa (%)', 'C_dt_min': 'Passo temporale suggerito (s)', 'nodi_predetti': 'Nodi predetti'})
    st.dataframe(X.round(2), hide_index=True, width='stretch'); ui.callout("Le mesh strutturate richiedono per le grandezze elettriche dimensioni molto minori o non raggiungono la soglia ai livelli provati; sulle mesh conformi la soglia è raggiunta in tutti i casi.")

# ------------------------------------------------------------------------------------------------ misure
with tab[5]:
    misure = sc.prove_corrispondenti(r, V, cf)
    if not misure: st.info("Non ci sono prove sperimentali per questa combinazione (non esistono prove a 30 V con la ricetta C, senza sale).")
    else:
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=[f"P{i + 1} ({'polpetta' if i in cp.SFERE[cf] else 'purea'})" for i in range(3)])
        for p in range(3):
            for j, (tt, TT) in enumerate(misure): fig.add_trace(go.Scatter(x=tt, y=TT[:, p] - TT[0, p], mode='markers', marker=dict(size=4, color='#495057', opacity=0.55), name=f'misure ({len(misure)} prove)', legendgroup='m', showlegend=(p == 0 and j == 0)), row=1, col=p + 1)
        st.caption(f"{len(misure)} prove con queste condizioni; temperature iniziali: " + ', '.join(f"{TT[0, :].mean():.1f} °C" for _, TT in misure) + ".")
        if st.button("▶ Aggiungi il modello nominale", key="do_nom"):
            with st.spinner("Simulazione…"):
                T0m = float(np.mean([TT[0, :].mean() for _, TT in misure])); fam = mo.famiglia_default(); ris = mo.simula(r, V, cf, fam, 1, T0=T0m, dt=10.0, durata=270.0)['serie']
                for p in range(3): fig.add_trace(go.Scatter(x=ris.t, y=ris[f'T_P{p + 1}'] - ris[f'T_P{p + 1}'].iloc[0], mode='lines', line=dict(color=ui.COL['caldo'], width=3), name='modello con parametri nominali', legendgroup='n', showlegend=(p == 0)), row=1, col=p + 1)
        fig.update_xaxes(title='Tempo (s)'); fig.update_yaxes(title='ΔT (K)', col=1); fig.update_layout(height=430, legend=dict(orientation='h', y=-0.22)); st.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------------------------------------------ scheda
with tab[6]:
    es = D.groupby('Famiglia').Esito_GCI.value_counts().unstack().fillna(0).astype(int); md = [f"# Dossier del caso {cid}", "", f"- Ricetta {r} (sale {dati.SALE[r]:g}%), σ purea a 20 °C {dati.sigma_T(r, 20):.2f} S/m, contrasto {dati.CONTR[r]:.2f}", f"- Tensione {V} V, configurazione {cf} ({dati.NPOLP[cf]} polpette)"]
    if rc is not None: md.append(f"- Riscaldamento massimo {rc.dT_max_tfix:.1f} K a {rc.t_valutazione:.0f} s; passo temporale suggerito {rc.C_dt_min:.0f} s")
    md += ["", "## Esiti di convergenza (GCI)", es.to_markdown(), "", "## Grandezze con incertezza sopra soglia"]
    for fam in ('strutturata', 'conforme'): md.append(f"- {fam}: " + (', '.join(dati.Q[q][0] for q in D[(D.Famiglia == fam) & (D.Esito_GCI == 'Sopra soglia')].Grandezza) or 'nessuna'))
    st.markdown('\n'.join(md)); st.download_button("Scarica la scheda (Markdown)", '\n'.join(md), f"dossier_{cid}.md", "text/markdown", key="do_dl")
ui.piede()
