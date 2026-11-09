"""Calibrazione dei parametri con simulazioni dirette sulle prove sperimentali."""
import time, json
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import ui, dati, calibra
from core import motore as mo
import calibrazione_protocollo as cp

ui.applica_tema()
ui.hero("Calibrazione dei parametri", "Stima dei parametri del modello sulle prove sperimentali con simulazioni dirette (nessun metamodello): le prove con le stesse condizioni iniziali condividono una simulazione, si confrontano gli incrementi di temperatura alle tre sonde pesati con l'incertezza di misura.", ["incrementi ΔT(t) alle sonde", "minimi quadrati pesati con prior", "modelli annidati M1 ⊂ M2 ⊂ M3"])
S = st.session_state; S.setdefault('risultati', []); S.setdefault('cal', None)
ud = calibra.uD_stima(1.5); tol = 2.0
sec = st.tabs(["1 · Impostazione e calibrazione", "2 · Risultati", "3 · Importa dal notebook di Colab"])

# ------------------------------------------------------------------------------------------------ impostazione
with sec[0]:
    G_all = calibra.gruppi_lista(tol); a, b = st.columns([1.1, 1])
    with a:
        ui.sezione("Dati")
        c = st.columns(3); fr = c[0].multiselect("Ricette", dati.RICS, default=['E', 'A'], key="ca_r"); fv = c[1].multiselect("Tensioni (V)", list(dati.VS), default=[30, 40, 50], key="ca_v"); fc = c[2].multiselect("Configurazioni", dati.CFG, default=['Tris', 'Middle'], key="ca_c")
        G = [g for g in G_all if g['ric'] in fr and int(g['V']) in fv and g['cfg'] in fc]; nr = sum(len(g['runs']) for g in G); k = st.columns(4)
        ui.kpi(k[0], f"{len(G_all)}", "gruppi totali", f"{len(calibra.prove_lista())} prove, tolleranza T₀ {tol:g} K"); ui.kpi(k[1], f"{len(G)}", "gruppi selezionati", f"{nr} prove"); ui.kpi(k[2], f"{len({g['cond'] for g in G})}", "condizioni"); ui.kpi(k[3], f"{ud['a']:.3f} · {ud['b']:.2f} K", "rumore di misura a · b", "u_D = √((aΔT)² + b²)")
        if G:
            M = pd.DataFrame([(g['ric'], int(g['V'])) for g in G], columns=['ric', 'V']).groupby(['ric', 'V']).size().unstack().reindex(dati.RICS).dropna(how='all'); fig = go.Figure(go.Heatmap(z=M.values, x=[f"{v} V" for v in M.columns], y=M.index, colorscale='Blues', text=M.values, texttemplate='%{text}', colorbar=dict(title='gruppi')))
            ui.layout(fig, "Gruppi di simulazione per ricetta e tensione", 280, yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
    with b:
        ui.sezione("Modello")
        mod = st.radio("Parametri da stimare", list(calibra.MODELLI), format_func=lambda m: f"{m} · {calibra.MODELLI[m][0]}", key="ca_mod"); nomi = calibra.MODELLI[mod][1]
        st.latex(r"\sigma'_m(T)=k_\sigma\,\sigma_{20}\,\big[1+k_\alpha\,\alpha\,(T-20)\big]")
        pr = pd.DataFrame([{'Parametro': calibra.PARAM_TXT[p], 'Prior (media)': cp.PRIOR[p][0], 'Prior (scarto tipo)': cp.PRIOR[p][1], 'Limiti': f"{cp.LIMITI[p][0]} – {cp.LIMITI[p][1]}"} for p in nomi]); st.dataframe(pr, hide_index=True, width='stretch'); st.caption("Prior e limiti sono assunti, non derivati da misure: il prior regolarizza la stima dei parametri poco identificabili.")
    ui.sezione("Divisione dei dati e simulatore")
    c = st.columns([1.2, 1, 1, 1, 1]); split = c[0].radio("Divisione addestramento / test", ["Fold per condizione", "Fuori campione per tensione", "Nessun test"], key="ca_split")
    fams = ['strutturata'] + (['conforme'] if mo.GMSH else []); fam = c[1].selectbox("Mesh", fams, index=len(fams) - 1, format_func=lambda f: mo.FAMIGLIE[f], key="ca_fam"); liv = c[2].select_slider("Livello", options=[0, 1], value=0, format_func=lambda k: f"{k} · {mo.NODI_LIVELLO[k]:,} nodi".replace(',', ' '), key="ca_liv"); dt = c[3].select_slider("Passo (s)", options=[10.0, 20.0, 30.0], value=20.0, key="ca_dt"); Ta = c[4].number_input("T aria (°C)", 15.0, 30.0, 21.0, 0.5, key="ca_Ta")
    train, test = list(G), []
    if split.startswith("Fold") and len(G) >= 4:
        K = st.slider("Numero di fold", 2, 6, 5, key="ca_K"); folds = cp.folds_per_condizione(G, K); fo = st.selectbox("Fold di test", list(range(K)), key="ca_fold"); train = [g for g in G if folds[g['cond']] != fo]; test = [g for g in G if folds[g['cond']] == fo]
    elif split.startswith("Fuori") and len(set(int(g['V']) for g in G)) >= 2:
        vt = st.selectbox("Tensione di test (V)", sorted({int(g['V']) for g in G}), index=len(sorted({int(g['V']) for g in G})) // 2, key="ca_vt"); train = [g for g in G if int(g['V']) != vt]; test = [g for g in G if int(g['V']) == vt]
    c = st.columns(3); ntr = c[0].slider("Gruppi di addestramento (massimo)", 2, 40, 8, key="ca_ntr"); nte = c[1].slider("Gruppi di test (massimo)", 0, 30, 6, key="ca_nte"); nfev = c[2].slider("Iterazioni massime", 2, 10, 4, key="ca_nfev")
    rng = np.random.default_rng(20261104)
    if len(train) > ntr: train = [train[i] for i in sorted(rng.choice(len(train), ntr, replace=False))]
    if len(test) > nte: test = [test[i] for i in sorted(rng.choice(len(test), nte, replace=False))] if nte else []
    tsim = {0: 0.5, 1: 2.0}[liv] * (10.0 / dt) * (270 / 270); n_eval = nfev * (len(nomi) + 1) * 1.3; costo = (n_eval * len(train) + 2 * (len(train) + len(test))) * tsim
    st.markdown(f"**Addestramento:** {len(train)} gruppi · **test:** {len(test)} gruppi · **tempo stimato:** circa {costo / 60:.1f} minuti (≈ {int(n_eval * len(train))} simulazioni, 270 s ciascuna).")
    ui.callout("Questa calibrazione in diretta è un'<b>anteprima</b> con mesh grossolana e pochi gruppi. La calibrazione completa (tutti i gruppi, mesh del protocollo, validazione incrociata) si esegue con il notebook di Colab, e i risultati si importano nella scheda 3.", 'warn')
    if st.button("▶ Avvia la calibrazione", type="primary", disabled=not train, key="ca_go"):
        sim = calibra.nuovo_simulatore(fam, liv, dt, Ta); barra = st.progress(0.0, text="Calibrazione in corso…"); cont = {'n': 0}; ph = st.empty(); trace = []
        def cb(th, costo_):
            cont['n'] += 1; trace.append(dict(valutazione=cont['n'], costo=costo_, **th)); barra.progress(min(0.99, cont['n'] / (nfev * (len(nomi) + 1) + 1)), text=f"Calibrazione in corso… valutazione {cont['n']}, costo {costo_:.1f}"); ph.dataframe(pd.DataFrame(trace).round(4), hide_index=True, width='stretch')
        try:
            r, pred = calibra.esegui_divisione(train, test, nomi, sim, ud, max_nfev=nfev, callback=cb, etichetta=mod); r['famiglia'], r['livello'], r['dt'], r['modello'] = fam, liv, dt, mod
            S.cal = dict(r=r, pred=pred, train=train, test=test, ud=ud, sim=sim, traccia=pd.DataFrame(trace)); S.risultati.append(dict(origine='in diretta', r=r, pred=pred)); barra.empty(); ph.empty(); st.success(f"Calibrazione completata in {r['minuti']:.1f} minuti: vedi la scheda «2 · Risultati».")
        except Exception as e: barra.empty(); st.error(f"Calibrazione non riuscita: {e}")

# ------------------------------------------------------------------------------------------------ risultati
with sec[1]:
    C = S.cal
    if C is None: st.info("Esegui una calibrazione nella scheda 1 per vedere i risultati.")
    else:
        r, pred = C['r'], C['pred']; k = st.columns(5); nom_tr = r['train_nominale']['RMSE']; ui.kpi(k[0], f"{r['train']['RMSE']:.2f} K", "RMSE addestramento", f"nominale {nom_tr:.2f} K")
        if 'test' in r: ui.kpi(k[1], f"{r['test']['RMSE']:.2f} K", "RMSE test", f"nominale {r['test_nominale']['RMSE']:.2f} K"); ui.kpi(k[2], f"{100 * (1 - r['test']['RMSE'] / r['test_nominale']['RMSE']):.0f}%", "riduzione dell'errore sul test", "rispetto ai parametri nominali")
        else: ui.kpi(k[1], "—", "RMSE test", "nessun test"); ui.kpi(k[2], f"{100 * (1 - r['train']['RMSE'] / nom_tr):.0f}%", "riduzione dell'errore", "sull'addestramento")
        ui.kpi(k[3], f"{r['rumore_min_train']:.2f} K", "rumore di misura (rms)", "limite inferiore atteso dell'RMSE"); ui.kpi(k[4], f"{r['nfev']} · {r['minuti']:.1f} min", "valutazioni · tempo", f"{r['n_sim']} simulazioni")
        ui.sezione("Parametri stimati"); rows = [];
        for p in r['theta']: m, s_ = cp.PRIOR[p]; rows.append({'Parametro': calibra.PARAM_TXT[p], 'Stima': r['theta'][p], 'Scarto tipo': r['sd'][p], 'Intervallo 95% basso': r['theta'][p] - 1.96 * r['sd'][p], 'Intervallo 95% alto': r['theta'][p] + 1.96 * r['sd'][p], 'Prior': f"{m} ± {s_}", 'Scarto dal prior (in σ)': (r['theta'][p] - m) / s_})
        a, b = st.columns([1.2, 1]); a.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width='stretch'); fig = go.Figure()
        for p in r['theta']: m, s_ = cp.PRIOR[p]; fig.add_shape(type='rect', x0=m - s_, x1=m + s_, y0=list(r['theta']).index(p) - 0.35, y1=list(r['theta']).index(p) + 0.35, fillcolor='rgba(134,142,150,0.25)', line=dict(width=0))
        fig.add_trace(go.Scatter(x=list(r['theta'].values()), y=list(range(len(r['theta']))), mode='markers', marker=dict(size=13, color=ui.COL['caldo']), error_x=dict(type='data', array=[1.96 * v for v in r['sd'].values()], color=ui.COL['caldo'], thickness=2.5), name='Stima'))
        fig.add_vline(x=1.0, line=dict(color='black', dash='dot')); ui.layout(fig, "Stima (intervallo 95%) e prior (banda grigia)", 300, yaxis=dict(tickvals=list(range(len(r['theta']))), ticktext=[p for p in r['theta']], autorange='reversed'), xaxis_title='Moltiplicatore', showlegend=False); b.plotly_chart(fig, width='stretch')
        if r.get('cov') is not None:
            ui.sezione("Correlazione e incertezza congiunta dei parametri"); cov = np.array(r['cov']); sdv = np.sqrt(np.diag(cov)); corr = cov / np.outer(sdv, sdv); nomi_ = r.get('nomi') or list(r['theta']); a, b = st.columns([1, 1])
            fig = go.Figure(go.Heatmap(z=corr, x=nomi_, y=nomi_, zmin=-1, zmax=1, colorscale='RdBu_r', text=np.round(corr, 2), texttemplate='%{text}', colorbar=dict(title='corr.'))); ui.layout(fig, "Correlazione fra le stime", 360, yaxis=dict(autorange='reversed')); a.plotly_chart(fig, width='stretch')
            if len(nomi_) >= 2:
                i_ = b.selectbox("Parametro sull'asse x", nomi_, index=0, key="ca_ex"); j_ = b.selectbox("Parametro sull'asse y", nomi_, index=1, key="ca_ey"); ii, jj = nomi_.index(i_), nomi_.index(j_)
                if ii != jj:
                    C2 = cov[np.ix_([ii, jj], [ii, jj])]; w, V_ = np.linalg.eigh(C2); w = np.clip(w, 1e-12, None); th_ = np.linspace(0, 2 * np.pi, 120); E = (V_ @ (np.sqrt(w * 5.991)[:, None] * np.vstack([np.cos(th_), np.sin(th_)]))).T + np.array([r['theta'][i_], r['theta'][j_]])
                    fig = go.Figure(go.Scatter(x=E[:, 0], y=E[:, 1], mode='lines', line=dict(color=ui.COL['caldo'], width=3), name='ellisse 95%')); fig.add_trace(go.Scatter(x=[r['theta'][i_]], y=[r['theta'][j_]], mode='markers', marker=dict(size=12, color=ui.COL['caldo']), name='stima')); fig.add_trace(go.Scatter(x=[1.0], y=[1.0], mode='markers', marker=dict(size=10, symbol='x', color='black'), name='nominale')); ui.layout(fig, f"Regione di confidenza al 95%: {i_} e {j_}", 330, xaxis_title=i_, yaxis_title=j_, legend=dict(orientation='h', y=-0.25)); b.plotly_chart(fig, width='stretch')
            forte = [(nomi_[i], nomi_[j], corr[i, j]) for i in range(len(nomi_)) for j in range(i + 1, len(nomi_)) if abs(corr[i, j]) > 0.8]
            if forte: ui.callout("Stime fortemente correlate: " + ", ".join(f"{x} e {y} ({c_:+.2f})" for x, y, c_ in forte) + ". Un singolo parametro non è identificabile separatamente: l'incertezza di ciascuno, presa da sola, sottostima quella della combinazione.", 'warn')
            else: ui.callout("Nessuna coppia di stime ha correlazione maggiore di 0.8 in valore assoluto: i parametri sono distinguibili con questi dati.")
        zz = pred[pred.set.isin(['train', 'test'])].copy(); zz['z'] = (zz.S - zz.D) / zz.U
        if len(zz) > 20:
            ui.sezione("Diagnostica dei residui standardizzati"); a, b, c_ = st.columns(3); from scipy import stats as _st; osm, osr = _st.probplot(zz.z.dropna().values, dist='norm', fit=False); lim = [float(min(osm.min(), osr.min())), float(max(osm.max(), osr.max()))]
            fig = go.Figure(go.Scattergl(x=osm, y=osr, mode='markers', marker=dict(size=4, color=ui.COL['strutt'], opacity=0.5))); fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines', line=dict(color='black', dash='dash'))); ui.layout(fig, "Grafico quantile-quantile", 340, xaxis_title='Quantili della normale', yaxis_title='Residuo standardizzato', showlegend=False); a.plotly_chart(fig, width='stretch')
            fig = go.Figure()
            for f_, col in (('polpetta', ui.COL['conf']), ('purea', ui.COL['strutt'])): fig.add_trace(go.Box(y=zz[zz.fase == f_].z, name=f_, marker_color=col, boxpoints=False))
            fig.add_hline(y=0, line=dict(color='black', width=1)); ui.layout(fig, "Residui per fase della sonda", 340, yaxis_title='(S − D) / u_D', showlegend=False); b.plotly_chart(fig, width='stretch')
            fig = go.Figure()
            for r_ in dati.RICS:
                if (zz.ric == r_).any(): fig.add_trace(go.Box(y=zz[zz.ric == r_].z, name=r_, marker_color=ui.COL_RIC[dati.RICS.index(r_)], boxpoints=False))
            fig.add_hline(y=0, line=dict(color='black', width=1)); ui.layout(fig, "Residui per ricetta", 340, yaxis_title='(S − D) / u_D', showlegend=False); c_.plotly_chart(fig, width='stretch')
            st.caption("Un residuo sistematicamente diverso da zero per una fase o una ricetta indica un errore di modello strutturale, che la calibrazione dei soli parametri globali non elimina.")
        ui.sezione("Confronto con le misure"); a, b = st.columns(2); pp = pred[pred.set.isin(['train', 'test'])]; fig = go.Figure()
        for sname, col in (('train', ui.COL['strutt']), ('test', ui.COL['conf'])):
            x = pp[pp.set == sname]
            if len(x): fig.add_trace(go.Scatter(x=x.D, y=x.S, mode='markers', marker=dict(size=5, color=col, opacity=0.5), name={'train': 'Addestramento', 'test': 'Test'}[sname]))
        lim = [float(pp[['D', 'S']].min().min()), float(pp[['D', 'S']].max().max())]; fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines', line=dict(color='black', dash='dash'), name='identità')); ui.layout(fig, "Simulato contro misurato (incrementi, K)", 400, xaxis_title='Misurato (K)', yaxis_title='Simulato (K)', legend=dict(orientation='h', y=-0.2)); a.plotly_chart(fig, width='stretch')
        pp = pp.assign(res=pp.S - pp.D); fig = go.Figure()
        for sname, col in (('train', ui.COL['strutt']), ('test', ui.COL['conf'])):
            x = pp[pp.set == sname]
            if len(x): g = x.groupby('t').res.agg(['mean', 'std']).reset_index(); fig.add_trace(go.Scatter(x=g.t, y=g['mean'], mode='lines', line=dict(color=col, width=3), name={'train': 'Addestramento', 'test': 'Test'}[sname])); fig.add_trace(go.Scatter(x=np.r_[g.t, g.t[::-1]], y=np.r_[g['mean'] + g['std'].fillna(0), (g['mean'] - g['std'].fillna(0))[::-1]], fill='toself', fillcolor='rgba(100,100,100,0.12)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_hline(y=0, line=dict(color='black', width=1)); ui.layout(fig, "Residuo (simulato − misurato) in funzione del tempo", 400, xaxis_title='Tempo (s)', yaxis_title='K', legend=dict(orientation='h', y=-0.2)); b.plotly_chart(fig, width='stretch')
        ui.sezione("Curve per condizione")
        gr_all = C['train'] + C['test']; ids = [g['id'] for g in gr_all]; gsel = st.selectbox("Gruppo", ids, format_func=lambda i: f"{i} · {'test' if i in [g['id'] for g in C['test']] else 'addestramento'}", key="ca_gsel"); g = next(x for x in gr_all if x['id'] == gsel); sim = C['sim']; sn = sim.calcola([g], {})[0]; sc = sim.calcola([g], r['theta'])[0]; fig = make_subplots(rows=1, cols=3, subplot_titles=[f"P{i + 1} ({'polpetta' if i in cp.SFERE[g['cfg']] else 'purea'})" for i in range(3)], shared_yaxes=True, horizontal_spacing=0.03)
        for p in range(3):
            for j, rr in enumerate(g['runs']): fig.add_trace(go.Scatter(x=rr['t'], y=rr['dT'][:, p], mode='markers', marker=dict(size=4, color='#868E96', opacity=0.55), name='Misure', legendgroup='m', showlegend=(p == 0 and j == 0)), row=1, col=p + 1)
            fig.add_trace(go.Scatter(x=cp.TG, y=sn[:, p], mode='lines', line=dict(color='#1f77b4', dash='dash', width=2.5), name='Modello nominale', legendgroup='n', showlegend=(p == 0)), row=1, col=p + 1); fig.add_trace(go.Scatter(x=cp.TG, y=sc[:, p], mode='lines', line=dict(color='#D9480F', width=3), name='Modello calibrato', legendgroup='c', showlegend=(p == 0)), row=1, col=p + 1)
        fig.update_xaxes(title='Tempo (s)'); fig.update_yaxes(title='ΔT (K)', col=1); fig.update_layout(height=420, legend=dict(orientation='h', y=-0.22)); st.plotly_chart(fig, width='stretch')
        if len(C['traccia']): st.markdown("**Traccia della calibrazione**"); st.dataframe(C['traccia'].round(4), hide_index=True, width='stretch')
        st.download_button("Scarica i risultati (JSON)", json.dumps({k: v for k, v in r.items()}, default=str, indent=1), f"calibrazione_{r['cand']}.json", "application/json", key="ca_dlj"); st.download_button("Scarica le previsioni (CSV)", pred.to_csv(index=False), f"previsioni_{r['cand']}.csv", "text/csv", key="ca_dlc")

# ------------------------------------------------------------------------------------------------ importa
with sec[2]:
    ui.callout("Il notebook di Colab calibra e valida su tutti i dati e salva, per ogni modello e fold, un file <code>cv_&lt;modello&gt;_fold&lt;k&gt;.json</code> e un file <code>previsioni_&lt;modello&gt;_fold&lt;k&gt;.csv</code>. Caricali qui per analizzarli nella pagina <b>Validazione</b>.")
    fj = st.file_uploader("File JSON dei risultati (cv_*.json)", type=['json'], accept_multiple_files=True, key="ca_fj"); fcv = st.file_uploader("File CSV delle previsioni (previsioni_*.csv), facoltativi", type=['csv'], accept_multiple_files=True, key="ca_fc")
    if fj and st.button("Importa i file", type="primary", key="ca_imp"):
        csvs = {f.name: f for f in fcv}; n = 0
        for f in fj:
            try:
                r = json.load(f); nome = f.name.replace('cv_', 'previsioni_').replace('.json', '.csv'); pred = pd.read_csv(csvs[nome]) if nome in csvs else None
                if pred is not None and 'U' not in pred.columns: pred['U'] = calibra.uD(pred.D.abs().values, ud)
                S.risultati.append(dict(origine='importato: ' + f.name, r=r, pred=pred)); n += 1
            except Exception as e: st.error(f"{f.name}: {e}")
        st.success(f"Importati {n} file. Passa alla pagina «Validazione».")
    if S.risultati: st.markdown("**Risultati disponibili in questa sessione**"); st.dataframe(pd.DataFrame([{'Origine': x['origine'], 'Modello': x['r'].get('cand'), 'Fold': x['r'].get('fold', '—'), 'RMSE addestramento': x['r'].get('train', {}).get('RMSE'), 'RMSE test': x['r'].get('test', {}).get('RMSE'), 'Previsioni': 'sì' if x['pred'] is not None else 'no'} for x in S.risultati]).round(3), hide_index=True, width='stretch')
ui.piede()
