"""Scenari di processo, ottimizzazione della tensione e incertezza della previsione."""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import ui, dati, scenari as sc
from core import motore as mo

ui.applica_tema()
ui.hero("Scenari e incertezza della previsione", "Il gemello digitale usato come strumento di processo: confronto di condizioni operative, ricerca della tensione che raggiunge una temperatura obiettivo senza superare il limite del modello, e propagazione dell'incertezza dei parametri sulle temperature previste, con il confronto con le misure.", ["scenari what-if", "ottimizzazione della tensione", "propagazione a ipercubo latino", "coefficienti di regressione standardizzati"])
S = st.session_state; fams = ['strutturata'] + (['conforme'] if mo.GMSH else []); COL = ['#1f77b4', '#D9480F', '#2B8A3E', '#7048E8']
t1, t2, t3 = st.tabs(["1 · Confronto di scenari", "2 · Ottimizzazione della tensione", "3 · Incertezza della previsione"])


def impostazioni(chiave, dt_def=20.0):
    c = st.columns([1.5, 1, 1, 1, 1]); fam = c[0].selectbox("Mesh", fams, index=len(fams) - 1, format_func=lambda f: mo.FAMIGLIE[f], key=f"{chiave}_f"); liv = c[1].select_slider("Livello", [0, 1], value=0, format_func=lambda k: f"{k} · {mo.NODI_LIVELLO[k]:,} nodi".replace(',', ' '), key=f"{chiave}_l")
    dt = c[2].select_slider("Passo (s)", [10.0, 20.0, 30.0], value=dt_def, key=f"{chiave}_dt"); dur = c[3].select_slider("Durata (s)", [90, 120, 180, 270, 360], value=270, key=f"{chiave}_d"); Ta = c[4].number_input("T aria (°C)", 15.0, 30.0, 21.0, 0.5, key=f"{chiave}_ta")
    return fam, liv, dt, float(dur), Ta


# ------------------------------------------------------------------------------------------------ confronto di scenari
with t1:
    ui.sezione("Scenari da confrontare"); n = st.slider("Numero di scenari", 2, 4, 3, key="sc_n"); defs = [('E', 40, 'Tris'), ('E', 50, 'Tris'), ('A', 40, 'Tris'), ('C', 40, 'Middle')]; scen = []; cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            st.markdown(f"<div class='card'><h4><span style='color:{COL[i]}'>●</span> Scenario {i + 1}</h4></div>", unsafe_allow_html=True); r = st.selectbox("Ricetta", dati.RICS, index=dati.RICS.index(defs[i][0]), format_func=lambda x: f"{x} · sale {dati.SALE[x]:g}%", key=f"sc_r{i}")
            v = st.select_slider("Tensione (V)", [30, 35, 40, 45, 50], value=defs[i][1], key=f"sc_v{i}"); c_ = st.selectbox("Configurazione", dati.CFG, index=dati.CFG.index(defs[i][2]), key=f"sc_c{i}"); T0 = st.number_input("T iniziale (°C)", 5.0, 30.0, 20.0, 0.5, key=f"sc_t{i}"); scen.append(dict(ric=r, V=v, config=c_, T0=T0))
    fam, liv, dt, dur, Ta = impostazioni("sc"); target = st.slider("Temperatura obiettivo alla sonda più fredda (°C)", 30, 85, 40, key="sc_tg")
    if st.button("▶ Simula gli scenari", type="primary", key="sc_go"):
        ris = []; barra = st.progress(0.0, text="Simulazione…")
        for i, s_ in enumerate(scen): ris.append(dict(sc=s_, r=sc.simula_scenario(s_, fam, liv, dt, dur, Ta))); barra.progress((i + 1) / n, text=f"Scenario {i + 1} di {n}")
        barra.empty(); S.scen = dict(ris=ris, target=target, dur=dur)
    R = S.get('scen')
    if R:
        ui.sezione("Risultati"); fig = go.Figure(); righe = []
        for i, x in enumerate(R['ris']):
            se = x['r']['serie']; s_ = x['sc']; nome = f"{i + 1} · {s_['ric']} · {s_['V']} V · {s_['config']}"; Tp = np.vstack([se[f'T_P{j}'].values for j in (1, 2, 3)]); fr = int(np.argmin(Tp[:, -1]))
            fig.add_trace(go.Scatter(x=se.t, y=Tp[fr], mode='lines', line=dict(color=COL[i], width=3), name=nome + f" · sonda più fredda (P{fr + 1})")); fig.add_trace(go.Scatter(x=se.t, y=se.T_max, mode='lines', line=dict(color=COL[i], width=2, dash='dash'), name=nome + " · T_max"))
            d = sc.sintesi_serie(se, R['target']); righe.append({'Scenario': nome, 'T sonda più fredda (°C)': d['T_fredda'], 'T_max (°C)': d['T_max'], 'Margine dal limite (K)': d['margine_limite'], f"Tempo a {R['target']} °C (s)": d['t_target_s'], 'Energia (kJ)': d['E_kJ'], 'Potenza media (W)': d['P_media_W'], 'Dislivello T_max − fredda (K)': d['T_max'] - d['T_fredda']})
        fig.add_hline(y=R['target'], line=dict(color='#2B8A3E', dash='dot'), annotation_text="obiettivo"); fig.add_hline(y=sc.LIM_T, line=dict(color='#C92A2A', dash='dot'), annotation_text="limite del modello (90 °C)")
        ui.layout(fig, "Temperature previste: sonda più fredda (continua) e nodo più caldo (tratteggiata)", 470, xaxis_title='Tempo (s)', yaxis_title='°C', legend=dict(orientation='h', y=-0.25)); st.plotly_chart(fig, width='stretch')
        T = pd.DataFrame(righe); st.dataframe(T.round(2), hide_index=True, width='stretch'); k = st.columns(3); ok = T[f"Tempo a {R['target']} °C (s)"].notna()
        if ok.any():
            mi = int(T.loc[ok, f"Tempo a {R['target']} °C (s)"].idxmin()); ui.kpi(k[0], f"{T.loc[mi, f'Tempo a {R['target']} °C (s)']:.0f} s", "scenario più rapido", T.loc[mi, 'Scenario']); me_ = int(T.loc[ok, 'Energia (kJ)'].idxmin()); ui.kpi(k[1], f"{T.loc[me_, 'Energia (kJ)']:.1f} kJ", "energia minima", T.loc[me_, 'Scenario']); mu = int(T.loc[ok, 'Dislivello T_max − fredda (K)'].idxmin()); ui.kpi(k[2], f"{T.loc[mu, 'Dislivello T_max − fredda (K)']:.1f} K", "riscaldamento più uniforme", T.loc[mu, 'Scenario'])
        else: ui.callout("Nessuno scenario raggiunge la temperatura obiettivo nella durata simulata: aumentare la durata o la tensione, oppure ridurre l'obiettivo.", 'warn')
        ui.callout("La temperatura massima è la grandezza meno convergente del protocollo (incertezza fino a circa il 28% sulla ricetta C): va letta come indicazione del punto caldo, non come valore da confrontare con le misure.", 'warn'); st.download_button("Scarica gli indicatori (CSV)", T.to_csv(index=False), "scenari.csv", "text/csv", key="sc_dl")

# ------------------------------------------------------------------------------------------------ ottimizzazione della tensione
with t2:
    ui.callout("Si cerca la tensione più bassa che porta la sonda più fredda alla temperatura obiettivo entro il tempo massimo, senza che il nodo più caldo superi il limite del modello. Il tempo di riscaldamento scala come una potenza negativa della tensione, e la simulazione ne stima l'esponente.")
    c = st.columns(5); r = c[0].selectbox("Ricetta", dati.RICS, index=2, format_func=lambda x: f"{x} · sale {dati.SALE[x]:g}%", key="op_r"); cf = c[1].selectbox("Configurazione", dati.CFG, key="op_c"); tg = c[2].slider("Obiettivo alla sonda più fredda (°C)", 30, 85, 40, key="op_tg"); tm = c[3].slider("Tempo massimo (s)", 60, 360, 270, 30, key="op_tm"); lim = c[4].slider("Limite sul nodo più caldo (°C)", 60, 95, 90, key="op_lim")
    c = st.columns([1, 1, 3]); vmin, vmax = c[0].number_input("V minima", 20, 60, 30, 5, key="op_v0"), c[1].number_input("V massima", 25, 70, 60, 5, key="op_v1"); fam, liv, dt, dur, Ta = impostazioni("op")
    Vs = list(range(int(vmin), int(vmax) + 1, 5)); st.caption(f"Tensioni da simulare: {', '.join(map(str, Vs))} V ({len(Vs)} simulazioni, durata {max(dur, tm):.0f} s).")
    if st.button("▶ Cerca la tensione", type="primary", key="op_go", disabled=len(Vs) < 3):
        righe = []; barra = st.progress(0.0, text="Simulazione…")
        for i, V in enumerate(Vs):
            se = sc.simula_scenario(dict(ric=r, V=V, config=cf), fam, liv, dt, float(max(dur, tm)), Ta)['serie']; Tp = np.vstack([se[f'T_P{j}'].values for j in (1, 2, 3)]); fr = int(np.argmin(Tp[:, -1])); tt = sc.tempo_a_target(se.t.values, Tp[fr], tg)
            if np.isfinite(tt): Tm_t = float(np.interp(tt, se.t.values, se.T_max.values)); E_t = float(np.interp(tt, se.t.values, se.E_in.values)) / 1000
            else: Tm_t = E_t = np.nan
            righe.append(dict(V=V, t_target=tt, T_max_al_target=Tm_t, E_kJ=E_t, sonda_fredda=f'P{fr + 1}', t_limite=sc.tempo_a_target(se.t.values, se.T_max.values, lim))); barra.progress((i + 1) / len(Vs), text=f"{V} V")
        barra.empty(); S.opt = dict(T=pd.DataFrame(righe), tg=tg, tm=tm, lim=lim, r=r, cf=cf)
    O = S.get('opt')
    if O:
        T = O['T'].copy(); T['entro_tempo'] = T.t_target <= O['tm']; T['entro_limite'] = T.T_max_al_target <= O['lim']; T['ammissibile'] = T.entro_tempo & T.entro_limite; amm = T[T.ammissibile]
        ok = T.dropna(subset=['t_target']); n_exp = np.nan
        if len(ok) >= 3: n_exp = -np.polyfit(np.log(ok.V), np.log(ok.t_target), 1)[0]
        k = st.columns(4); ui.kpi(k[0], f"{amm.V.min()} V" if len(amm) else "nessuna", "tensione ammissibile minima", f"obiettivo {O['tg']} °C entro {O['tm']} s, T_max ≤ {O['lim']} °C"); ui.kpi(k[1], f"{amm.loc[amm.V.idxmin(), 't_target']:.0f} s" if len(amm) else "—", "tempo all'obiettivo", "alla tensione minima ammissibile"); ui.kpi(k[2], f"{n_exp:.2f}" if np.isfinite(n_exp) else "—", "esponente del tempo", "t ∝ V^(−n)"); ui.kpi(k[3], f"{int(T.ammissibile.sum())} su {len(T)}", "tensioni ammissibili", "")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Tempo per raggiungere l'obiettivo", "Nodo più caldo e energia all'obiettivo"), specs=[[{}, {'secondary_y': True}]]); fig.add_trace(go.Scatter(x=T.V, y=T.t_target, mode='lines+markers', marker=dict(size=10, color=ui.COL['caldo']), line=dict(color=ui.COL['caldo'], width=3), name='tempo (s)'), row=1, col=1); fig.add_hline(y=O['tm'], line=dict(color='black', dash='dot'), row=1, col=1)
        fig.add_trace(go.Scatter(x=T.V, y=T.T_max_al_target, mode='lines+markers', line=dict(color='#C92A2A', width=3), marker=dict(size=9), name='T_max (°C)'), row=1, col=2, secondary_y=False); fig.add_hline(y=O['lim'], line=dict(color='#C92A2A', dash='dot'), row=1, col=2); fig.add_trace(go.Scatter(x=T.V, y=T.E_kJ, mode='lines+markers', line=dict(color=ui.COL['strutt'], width=3, dash='dash'), marker=dict(size=9), name='energia (kJ)'), row=1, col=2, secondary_y=True)
        fig.update_xaxes(title='Tensione (V)'); fig.update_yaxes(title='s', type='log', row=1, col=1); fig.update_yaxes(title='°C', secondary_y=False, row=1, col=2); fig.update_yaxes(title='kJ', secondary_y=True, row=1, col=2); fig.update_layout(height=430, legend=dict(orientation='h', y=-0.22)); st.plotly_chart(fig, width='stretch')
        st.dataframe(T.round(2).rename(columns={'V': 'Tensione (V)', 't_target': 'Tempo all\'obiettivo (s)', 'T_max_al_target': 'T_max all\'obiettivo (°C)', 'E_kJ': 'Energia (kJ)', 'sonda_fredda': 'Sonda più fredda', 't_limite': 'Tempo al limite (s)'}), hide_index=True, width='stretch')

# ------------------------------------------------------------------------------------------------ incertezza della previsione
with t3:
    ui.callout("Gli intervalli di incertezza dei parametri (±10% su σ₂₀, ±41% su α, ±8% su f_c, da letteratura) sono propagati con un campione a ipercubo latino e simulazioni dirette: le bande mostrano quanto l'incertezza dei parametri, da sola, incide sulle temperature previste. Non comprendono l'incertezza numerica né quella di modello.")
    c = st.columns(4); r = c[0].selectbox("Ricetta", dati.RICS, index=2, format_func=lambda x: f"{x} · sale {dati.SALE[x]:g}%", key="un_r"); V = c[1].select_slider("Tensione (V)", [30, 35, 40, 45, 50], value=40, key="un_v"); cf = c[2].selectbox("Configurazione", dati.CFG, key="un_c"); T0 = c[3].number_input("T iniziale (°C)", 5.0, 30.0, 20.0, 0.5, key="un_t0")
    c = st.columns([2, 1, 1]); par = c[0].multiselect("Parametri incerti", dati.PAR, default=sc.PARAM_UNC, format_func=lambda p: dati.PAR_TXT[p] + " · " + dati.PAR_DESC[p], key="un_p"); N = c[1].slider("Simulazioni", 8, 60, 16, 4, key="un_n"); seed = c[2].number_input("Seme", 1, 9999, 1234, key="un_s")
    with st.expander("Semiampiezze relative degli intervalli (modificabili)"):
        cc = st.columns(min(len(par), 6)) if par else []; semi = {p: cc[i % len(cc)].number_input(dati.PAR_TXT[p], 0.01, 0.8, sc.SEMI[p], 0.01, key=f"un_se_{p}") for i, p in enumerate(par)}
    fam, liv, dt, dur, Ta = impostazioni("un"); target = st.slider("Obiettivo alla sonda più fredda (°C)", 30, 85, 38, key="un_tg")
    if par and st.button("▶ Propaga l'incertezza", type="primary", key="un_go"):
        M = sc.lhs_moltiplicatori(N, par, int(seed), semi); sce = dict(ric=r, V=V, config=cf, T0=T0); barra = st.progress(0.0, text="Simulazioni…")
        nom = sc.simula_scenario(sce, fam, liv, dt, dur, Ta)['serie']; tempi, T, Tm, E = sc.ensemble(sce, fam, liv, M, dt, dur, Ta, avanz=lambda x: barra.progress(x, text=f"{int(x * N)} di {N} simulazioni")); barra.empty(); S.ens = dict(M=M, tempi=tempi, T=T, Tm=Tm, E=E, nom=nom, sce=sce, target=target, par=par)
    X = S.get('ens')
    if X:
        t = X['tempi']; T, nom, sce = X['T'], X['nom'], X['sce']; fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=[f"P{i + 1} ({'polpetta' if i in __import__('calibrazione_protocollo').SFERE[sce['config']] else 'purea'})" for i in range(3)]); misure = sc.prove_corrispondenti(sce['ric'], sce['V'], sce['config']); mostra = st.toggle("Sovrapponi le misure (incrementi)", value=bool(misure), key="un_ms")
        for p in range(3):
            inc = T[:, p, :] - T[:, p, :1]; lo, me, hi = np.percentile(inc, [5, 50, 95], axis=0); fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]], fill='toself', fillcolor='rgba(217,72,15,0.20)', line=dict(width=0), name='banda 5-95%', legendgroup='b', showlegend=(p == 0), hoverinfo='skip'), row=1, col=p + 1)
            fig.add_trace(go.Scatter(x=t, y=me, mode='lines', line=dict(color=ui.COL['caldo'], width=3), name='mediana', legendgroup='m', showlegend=(p == 0)), row=1, col=p + 1); fig.add_trace(go.Scatter(x=nom.t, y=nom[f'T_P{p + 1}'] - nom[f'T_P{p + 1}'].iloc[0], mode='lines', line=dict(color='black', dash='dash', width=2), name='parametri nominali', legendgroup='n', showlegend=(p == 0)), row=1, col=p + 1)
            if mostra:
                for j, (tt, TT) in enumerate(misure): fig.add_trace(go.Scatter(x=tt, y=TT[:, p] - TT[0, p], mode='markers', marker=dict(size=4, color='#495057', opacity=0.5), name='misure', legendgroup='x', showlegend=(p == 0 and j == 0)), row=1, col=p + 1)
        fig.update_xaxes(title='Tempo (s)'); fig.update_yaxes(title='ΔT (K)', col=1); fig.update_layout(height=440, legend=dict(orientation='h', y=-0.22)); ui.sezione("Bande di previsione"); st.plotly_chart(fig, width='stretch')
        if mostra and not misure: st.caption("Non ci sono misure per questa combinazione di ricetta, tensione e configurazione.")
        elif mostra: st.caption(f"{len(misure)} prove con le stesse condizioni; le misure partono da temperature iniziali diverse da quella simulata, per cui si confrontano gli incrementi.")
        fin = pd.DataFrame({'T_P1 finale': T[:, 0, -1], 'T_P2 finale': T[:, 1, -1], 'T_P3 finale': T[:, 2, -1], 'T_max finale': X['Tm'][:, -1], 'Energia (kJ)': X['E'] / 1000}); fr = np.argmin(T[:, :, -1].mean(0)); tt_ = np.array([sc.tempo_a_target(t, T[i, fr], X['target']) for i in range(len(T))]); fin['Tempo all\'obiettivo (s)'] = tt_
        ui.sezione("Distribuzione degli esiti"); k = st.columns(4); sp = fin.describe(percentiles=[0.05, 0.5, 0.95]).T
        for j, (col, lab, u) in enumerate((('T_P1 finale', 'T alla sonda P1', '°C'), ('T_max finale', 'T_max', '°C'), ('Energia (kJ)', 'Energia', 'kJ'), ("Tempo all'obiettivo (s)", f"Tempo a {X['target']} °C", 's'))):
            if col in sp.index and np.isfinite(sp.loc[col, '50%']): ui.kpi(k[j], f"{sp.loc[col, '50%']:.1f} {u}", lab + " (mediana)", f"5-95%: {sp.loc[col, '5%']:.1f} – {sp.loc[col, '95%']:.1f}")
        a, b = st.columns(2); fig = go.Figure(); fig.add_histogram(x=fin['T_P1 finale'], nbinsx=12, marker_color=ui.COL['strutt'], name='T_P1'); ui.layout(fig, "Temperatura finale alla sonda P1", 340, xaxis_title='°C', yaxis_title='simulazioni', showlegend=False); a.plotly_chart(fig, width='stretch')
        src, r2 = sc.coefficienti_standardizzati(X['M'], fin['T_P1 finale'].values); fig = go.Figure(go.Bar(x=[src[p] for p in src], y=[dati.PAR_TXT[p] for p in src], orientation='h', marker_color=[ui.COL_PAR[p] for p in src])); ui.layout(fig, f"Contributo dei parametri a T_P1 finale (SRC, R² = {r2:.2f})", 340, xaxis_title='Coefficiente di regressione standardizzato', yaxis=dict(autorange='reversed')); b.plotly_chart(fig, width='stretch')
        st.dataframe(sp[['5%', '50%', '95%', 'mean', 'std']].round(2).rename(columns={'mean': 'media', 'std': 'scarto tipo'}), width='stretch'); cm = pd.concat([X['M'].reset_index(drop=True), fin], axis=1); st.download_button("Scarica il campione e gli esiti (CSV)", cm.to_csv(index=False), "incertezza_previsione.csv", "text/csv", key="un_dl")
ui.piede()
