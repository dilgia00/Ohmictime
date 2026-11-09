"""Panoramica: caso di studio, flusso degli studi e risultati principali."""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from core import ui, dati

ui.applica_tema()
ui.hero("Gemello digitale del riscaldamento ohmico multifase",
        "Modello elettro-termico 3D agli elementi finiti di una cella con purea e polpette: simulazione, verifica numerica, sensitività, calibrazione e validazione sui dati sperimentali, tutto sullo stesso caso di studio.",
        ["Cella 99 × 54 × 66 mm", "5 ricette · 5 configurazioni · 30, 40, 50 V", "75 casi", "2 famiglie di discretizzazione", "256 prove sperimentali"])
D = dati.convergenza(); R = dati.riepilogo(); X = dati.sperimentali()
ab = D[D.Studio.isin(['A', 'B'])]; sonde = D[(D.Studio == 'B') & D.Grandezza.isin(['T_P1', 'T_P2', 'T_P3'])]
perc = lambda d: 100 * (d.Esito_GCI != 'Sopra soglia').mean()
perc1 = lambda d: f"{perc(d):.1f}"
c = st.columns(5)
ui.kpi(c[0], "75", "casi simulati per famiglia", "5 ricette × 5 configurazioni × 3 tensioni")
ui.kpi(c[1], f"{perc1(sonde[sonde.Famiglia == 'conforme'])}% · {perc1(sonde[sonde.Famiglia == 'strutturata'])}%", "temperature di sonda sotto soglia", "mesh conformi · griglie strutturate")
ui.kpi(c[2], f"{perc1(ab[(ab.Famiglia == 'conforme')])}% · {perc1(ab[(ab.Famiglia == 'strutturata')])}%", "tutte le grandezze sotto soglia", "studi spaziali A e B")
ui.kpi(c[3], f"{X.run.nunique()}", "prove sperimentali", f"{X.groupby(['cfg', 'V', 'NaCl']).ngroups} condizioni, 5 s di campionamento")
ui.kpi(c[4], "9", "parametri analizzati", "sensitività locale e globale (Sobol)")

ui.sezione("Cosa si può fare in questa app")
r1 = st.columns(3)
ui.card(r1[0], "1 · Cella, ricette e dati", "Geometria 3D con le cinque configurazioni delle polpette, conducibilità delle ricette e delle polpette in funzione della temperatura, esplorazione delle prove sperimentali e del modello del rumore.")
ui.card(r1[1], "2 · Simulatore", "Simulazione 3D accoppiata di qualsiasi caso: curve alle sonde, campi di temperatura e di conducibilità, potenza e bilancio energetico, confronto con una prova sperimentale.")
ui.card(r1[2], "3 · Verifica numerica e incertezza", "Convergenza di mesh e di passo temporale con GCI e minimi quadrati, soglie per grandezza, confronto fra le due famiglie, e verifica rapida eseguita in diretta.")
r2 = st.columns(3)
ui.card(r2[0], "4 · Sensitività", "Locale (effetti per +10%, quote, fasi, esponente della tensione, identificabilità) e globale (Sobol), con confronto fra contrasti, tensioni e configurazioni, e calcolo locale in diretta.")
ui.card(r2[1], "5 · Calibrazione", "Stima dei parametri con simulazioni dirette sulle prove sperimentali, con raggruppamento per condizioni iniziali, incertezza di misura e modelli annidati.")
ui.card(r2[2], "6 · Validazione", "Valutazione fuori campione con RMSE, bias, MAE e R², confronto con il rumore di misura e analisi dei residui per ricetta, tensione, configurazione e sonda.")
r3 = st.columns(3)
ui.card(r3[0], "7 · Modello nominale e misure", "Verifica preliminare prima della calibrazione: il modello con i parametri nominali contro tutte le prove, con bias per ricetta, tensione, configurazione e fase delle sonde e confronto con il rumore di misura.")
ui.card(r3[1], "8 · Dossier del caso", "Per ogni combinazione di ricetta, tensione e configurazione: esiti di convergenza e incertezza di entrambe le famiglie, passo temporale, sensitività locale e globale, mesh suggerite, misure e scheda scaricabile.")
ui.card(r3[2], "9 · Scenari e incertezza", "Il gemello digitale come strumento di processo: confronto di scenari, tensione minima che raggiunge una temperatura obiettivo e propagazione dell'incertezza dei parametri con bande di previsione sulle temperature.")
st.write("")
nav = st.columns(6)
for col, (p, l) in zip(nav, (("pagine/cella_dati.py", "Cella e dati"), ("pagine/simulatore.py", "Simulatore"), ("pagine/verifica.py", "Verifica"), ("pagine/sensitivita.py", "Sensitività"), ("pagine/calibrazione.py", "Calibrazione"), ("pagine/validazione.py", "Validazione"))):
    try: col.page_link(p, label=l, width='stretch')
    except Exception: pass
nav2 = st.columns(6)
for col, (p, l) in zip(nav2, (("pagine/nominale.py", "Modello nominale"), ("pagine/dossier.py", "Dossier del caso"), ("pagine/scenari.py", "Scenari"), ("pagine/modello_base.py", "Modello base"), ("pagine/guida.py", "Guida"))):
    try: col.page_link(p, label=l, width='stretch')
    except Exception: pass

ui.sezione("Il caso di studio")
a, b = st.columns([1.15, 1])
with a:
    conf = st.radio("Configurazione delle polpette", dati.CFG, horizontal=True, format_func=lambda c: f"{c} ({dati.NPOLP[c]})", key="pan_conf")
    L, W, H = 99.0, 54.0, 66.0; xs = [17.25, 49.5, 81.75]
    fig = go.Figure(); fig.add_shape(type='rect', x0=0, y0=0, x1=L, y1=W, line=dict(color='#495057', width=2), fillcolor='#FFF4E6')
    for x in (0, L): fig.add_shape(type='rect', x0=x - 3 if x == 0 else x, y0=0, x1=x if x == 0 else x + 3, y1=W, line=dict(color='#868E96'), fillcolor='#ADB5BD')
    for i, x in enumerate(xs):
        occ = dati.OCC[conf][i] == 1
        fig.add_shape(type='circle', x0=x - 15, y0=W / 2 - 15, x1=x + 15, y1=W / 2 + 15, line=dict(color='#C2410C' if occ else '#CED4DA', width=2, dash='solid' if occ else 'dot'), fillcolor='rgba(217,72,15,0.35)' if occ else 'rgba(0,0,0,0)')
        fig.add_trace(go.Scatter(x=[x], y=[W / 2], mode='markers+text', marker=dict(size=11, color='#0B3954', symbol='x'), text=[f"P{i + 1}"], textposition='top center', showlegend=False, hovertemplate=f"Sonda P{i + 1}<br>{'in polpetta' if occ else 'in purea'}<extra></extra>"))
    fig.update_layout(height=330, xaxis=dict(range=[-8, L + 8], title='x (mm), tra gli elettrodi', scaleanchor='y', showgrid=False), yaxis=dict(range=[-4, W + 4], title='y (mm)', showgrid=False), margin=dict(l=50, r=10, t=10, b=45), plot_bgcolor='white')
    st.plotly_chart(fig, width='stretch', config=dict(displayModeBar=False))
    st.caption("Vista dall'alto: gli elettrodi coprono le facce estreme; le tre posizioni possibili delle sfere (R = 15 mm, spazio fra le sfere di 2.25 mm) e le sonde P1–P3 al centro di ciascuna; altezza della cella 66 mm.")
with b:
    t = pd.DataFrame({'Ricetta': dati.RICS, 'Sale (%)': [dati.SALE[r] for r in dati.RICS], 'σ₂₀ purea (S/m)': [round(20 * dati.AB[r][0] + dati.AB[r][1], 2) for r in dati.RICS], 'Contrasto σp/σm': [round(dati.CONTR[r], 2) for r in dati.RICS]})
    st.markdown("**Ricette (in ordine di contrasto crescente)**"); st.dataframe(t, hide_index=True, width='stretch')
    st.markdown("**Condizioni sperimentali**"); st.markdown("- Tensioni efficaci: **30, 40, 50 V**, 50 Hz (generatore BK Precision 9832B)\n- Polpette: conducibilità σ = 0.034·T + 0.609 S/m (Tab. 5.1 della tesi)\n- Sonde: termocoppie di tipo T al centro delle tre posizioni, 5 s di campionamento fino a 270 s")

ui.sezione("Flusso degli studi")
cols = st.columns(2)
with cols[0]:
    ui.passo(0, "<b>Famiglie di discretizzazione</b> armonizzate a pari nodi (800 – 51 200): griglia strutturata e mesh conforme")
    ui.passo(1, "<b>Verifica del codice</b> con soluzioni analitiche (riscaldamento adiabatico, sfera in campo uniforme)")
    ui.passo(2, "<b>Incertezza numerica</b> spaziale e temporale: GCI e minimi quadrati, tre esiti di convergenza, soglie per grandezza")
    ui.passo(3, "<b>Sensitività locale</b> su nove parametri, quote, fasi, identificabilità")
with cols[1]:
    ui.passo(4, "<b>Sensitività globale</b> (Morris e Sobol su metamodello) e concordanza con la locale")
    ui.passo(5, "<b>Confronto fra le due famiglie</b> sulle prestazioni, i limiti e il costo")
    ui.passo(6, "<b>Audit dei dati</b> sperimentali e modello del rumore di misura")
    ui.passo(7, "<b>Calibrazione e validazione</b> con simulazioni dirette e validazione incrociata")

ui.sezione("Risultati principali del protocollo")
g1, g2 = st.columns(2)
with g1:
    rows = []
    for st_, nome in (('A', 'Elettrico (A)'), ('B', 'Accoppiato (B)'), ('C', 'Temporale (C)')):
        for fam, fn in (('strutturata', 'Griglie strutturate'), ('conforme', 'Mesh conformi')): d = D[(D.Studio == st_) & (D.Famiglia == fam)]; rows.append((nome, fn, 100 * (d.Esito_GCI != 'Sopra soglia').mean()))
    t = pd.DataFrame(rows, columns=['Studio', 'Famiglia', 'v']); fig = go.Figure()
    for fn, col in (('Griglie strutturate', ui.COL['strutt']), ('Mesh conformi', ui.COL['conf'])): s = t[t.Famiglia == fn]; fig.add_bar(x=s.Studio, y=s.v, name=fn, marker_color=col, text=[f"{v:.0f}%" for v in s.v], textposition='outside')
    fig.update_layout(barmode='group', yaxis=dict(title='Casi con incertezza ≤ soglia (%)', range=[0, 112]), height=380, title=dict(text='Incertezza numerica sotto soglia (GCI)', x=0.01, font=dict(size=15)), legend=dict(orientation='h', y=-0.18)); st.plotly_chart(fig, width='stretch')
with g2:
    S = dati.sobol(); S = S[S.uscita == 'rms_sonde']; m = S.groupby(['ric', 'parametro']).ST.mean().unstack()[dati.PAR].loc[dati.RICS]
    fig = go.Figure(go.Heatmap(z=m.values, x=[dati.PAR_TXT[p] for p in dati.PAR], y=[f"{r} (c={dati.CONTR[r]:.2f})" for r in dati.RICS], colorscale='Viridis', zmin=0, zmax=0.7, text=np.round(m.values, 2), texttemplate='%{text}', colorbar=dict(title='S_T')))
    fig.update_layout(height=380, title=dict(text='Sensitività globale: indice totale di Sobol medio per ricetta', x=0.01, font=dict(size=15)), yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
ui.callout("<b>Come leggere i risultati.</b> «Sotto soglia» significa che l'incertezza numerica stimata non supera la soglia scelta per quella grandezza (2.5% per le grandezze legate alle misure, 5% per gli indicatori locali): descrive l'incertezza di discretizzazione e non l'errore del modello rispetto alle misure, che si valuta con la calibrazione e la validazione.")
ui.piede()
