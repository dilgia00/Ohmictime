"""Validazione: valutazione fuori campione, confronto con il rumore di misura, analisi dei residui e stabilità dei parametri."""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from core import ui, dati, calibra
from core import motore as mo
import calibrazione_protocollo as cp

ui.applica_tema()
ui.hero("Validazione fuori campione", "Valutazione dei parametri calibrati su condizioni non usate per la calibrazione: RMSE, bias, MAE e R² per ricetta, tensione, configurazione, fase e sonda, confronto con il rumore di misura e analisi dei residui.", ["RMSE · bias · MAE · R²", "validazione incrociata per condizione", "confronto con l'incertezza di misura"])
S = st.session_state; S.setdefault('risultati', []); S.setdefault('cal', None); ud = calibra.uD_stima(1.5)
RIS = S.risultati
tab_v, tab_n = st.tabs(["Analisi dei risultati", "Validazione in diretta su nuove condizioni"])

with tab_n:
    C = S.cal
    if C is None: st.info("Calibra prima un modello nella pagina «Calibrazione»: i parametri stimati saranno valutati qui su condizioni non usate per l'addestramento.")
    else:
        r = C['r']; th = r['theta']; st.markdown(f"**Parametri da validare:** " + ", ".join(f"{k} = {v:.3f}" for k, v in th.items()) + f" · modello {r['cand']} · mesh {mo.FAMIGLIE[r['famiglia']]} livello {r['livello']}")
        G_all = calibra.gruppi_lista(2.0); gia = {g['id'] for g in C['train']}; libere = [g for g in G_all if g['id'] not in gia]; c = st.columns(4)
        fr = c[0].multiselect("Ricette", dati.RICS, default=dati.RICS, key="vn_r"); fv = c[1].multiselect("Tensioni (V)", list(dati.VS), default=[40], key="vn_v"); fc = c[2].multiselect("Configurazioni", dati.CFG, default=dati.CFG, key="vn_c"); nmax = c[3].slider("Gruppi (massimo)", 2, 30, 8, key="vn_n")
        pool = [g for g in libere if g['ric'] in fr and int(g['V']) in fv and g['cfg'] in fc]; rng = np.random.default_rng(7); sel = [pool[i] for i in sorted(rng.choice(len(pool), min(nmax, len(pool)), replace=False))] if pool else []
        st.caption(f"{len(pool)} gruppi disponibili non usati per l'addestramento; se ne valutano {len(sel)}. Tempo stimato: circa {2 * len(sel) * {0: 0.5, 1: 2.0}[r['livello']] * (10.0 / r['dt']):.0f} s.")
        if st.button("▶ Valida su queste condizioni", type="primary", disabled=not sel, key="vn_go"):
            with st.spinner("Simulazioni in corso…"):
                pred = calibra.valuta_parametri(sel, th, C['sim'], ud); t_ = pred[pred.set == 'test']; n_ = pred[pred.set == 'test_nominale']; rr = dict(cand=r['cand'], theta=th, sd=r['sd'], gruppi_test=len(sel), test=cp.indici(t_.S, t_.D), test_nominale=cp.indici(n_.S, n_.D), train=r['train'], rumore_min_test=float(np.sqrt(np.mean(t_.U ** 2))), condizioni_test=sorted({g['cond'] for g in sel}), nota='validazione in diretta')
                S.risultati.append(dict(origine='validazione in diretta', r=rr, pred=pred)); st.success("Validazione completata: i risultati sono nella scheda «Analisi dei risultati».")

with tab_v:
    if not RIS: st.info("Nessun risultato disponibile. Esegui una calibrazione (pagina «Calibrazione») o importa i file del notebook di Colab.")
    else:
        nomi = [f"{i + 1} · {x['origine']} · {x['r'].get('cand', '?')}" + (f" · fold {x['r']['fold']}" if 'fold' in x['r'] else '') for i, x in enumerate(RIS)]
        sel = st.multiselect("Risultati da analizzare", nomi, default=nomi[-1:], key="vv_sel"); idx = [nomi.index(n) for n in sel]; R = [RIS[i] for i in idx]
        if not R: st.stop()
        rows = []
        for i, x in zip(idx, R):
            r = x['r']; te = r.get('test'); rows.append({'Risultato': i + 1, 'Origine': x['origine'], 'Modello': r.get('cand'), 'Fold': r.get('fold', '—'), 'RMSE addestramento (K)': r.get('train', {}).get('RMSE'), 'RMSE test (K)': te['RMSE'] if te else np.nan, 'RMSE test nominale (K)': r.get('test_nominale', {}).get('RMSE', np.nan), 'Rumore di misura rms (K)': r.get('rumore_min_test', r.get('rumore_min_train')), 'Bias test (K)': te['bias'] if te else np.nan, 'MAE test (K)': te['MAE'] if te else np.nan, 'R² test': te['R2'] if te else np.nan, 'Gruppi test': r.get('gruppi_test')})
        T = pd.DataFrame(rows); T['Divario test − addestramento (K)'] = T['RMSE test (K)'] - T['RMSE addestramento (K)']; T['Riduzione vs nominale (%)'] = 100 * (1 - T['RMSE test (K)'] / T['RMSE test nominale (K)']); T['RMSE test / rumore'] = T['RMSE test (K)'] / T['Rumore di misura rms (K)']
        k = st.columns(4); tt = T.dropna(subset=['RMSE test (K)'])
        if len(tt): ui.kpi(k[0], f"{tt['RMSE test (K)'].mean():.2f} K", "RMSE test (media)", f"nominale {tt['RMSE test nominale (K)'].mean():.2f} K"); ui.kpi(k[1], f"{tt['Riduzione vs nominale (%)'].mean():.0f}%", "riduzione dell'errore", "rispetto ai parametri nominali"); ui.kpi(k[2], f"{tt['RMSE test / rumore'].mean():.1f}×", "RMSE test / rumore di misura", "1 = limite del rumore"); ui.kpi(k[3], f"{tt['Divario test − addestramento (K)'].mean():+.2f} K", "divario test − addestramento", "indicatore di sovradattamento")
        st.dataframe(T.round(3), hide_index=True, width='stretch')
        if len(tt):
            fig = go.Figure()
            for col, nm, c in (('RMSE addestramento (K)', 'Addestramento', '#1f77b4'), ('RMSE test (K)', 'Test (calibrato)', '#D9480F'), ('RMSE test nominale (K)', 'Test (nominale)', '#868E96'), ('Rumore di misura rms (K)', 'Rumore di misura', '#2B8A3E')): fig.add_bar(x=[f"{a} · {b} · {c_}" for a, b, c_ in zip(T['Risultato'], T['Modello'], T['Fold'])], y=T[col], name=nm, marker_color=c)
            ui.layout(fig, "RMSE per risultato", 380, barmode='group', yaxis_title='K', legend=dict(orientation='h', y=-0.2)); st.plotly_chart(fig, width='stretch')
        P = [x['pred'].assign(res_id=i + 1) for i, x in zip(idx, R) if x['pred'] is not None]
        if not P: st.caption("Questi risultati non includono le previsioni (file previsioni_*.csv): le analisi per gruppo richiedono le previsioni.")
        else:
            P = pd.concat(P, ignore_index=True); 
            if 'U' not in P.columns: P['U'] = calibra.uD(P.D.abs().values, ud)
            te = P[P.set == 'test'].copy(); no = P[P.set == 'test_nominale'].copy()
            if te.empty: te = P[P.set == 'train'].copy(); st.caption("Nessun insieme di test: si analizzano le previsioni di addestramento.")
            te['res'] = te.S - te.D; te['z'] = te.res / te.U; ui.sezione("Prestazioni per gruppo")
            gr = st.radio("Raggruppa per", ['ric', 'V', 'cfg', 'fase', 'sonda', 'ric × V'], horizontal=True, format_func=lambda x: {'ric': 'ricetta', 'V': 'tensione', 'cfg': 'configurazione', 'fase': 'fase', 'sonda': 'sonda', 'ric × V': 'ricetta × tensione'}[x], key="vv_gr")
            def metr(d): m = mo.indici_confronto(d.S, d.D); m['entro_u'] = 100 * float((d.res.abs() <= d.U).mean()); m['entro_2u'] = 100 * float((d.res.abs() <= 2 * d.U).mean()); return pd.Series(m)
            if gr == 'ric × V':
                M = te.groupby(['ric', 'V']).apply(lambda d: np.sqrt((d.res ** 2).mean()), include_groups=False).unstack().reindex(dati.RICS).dropna(how='all'); fig = go.Figure(go.Heatmap(z=M.values, x=[f"{v:g} V" for v in M.columns], y=M.index, colorscale='YlOrRd', text=np.round(M.values, 2), texttemplate='%{text}', colorbar=dict(title='RMSE (K)')))
                ui.layout(fig, "RMSE sul test per ricetta e tensione", 380, yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
                Mb = te.groupby(['ric', 'V']).res.mean().unstack().reindex(dati.RICS).dropna(how='all'); fig = go.Figure(go.Heatmap(z=Mb.values, x=[f"{v:g} V" for v in Mb.columns], y=Mb.index, colorscale='RdBu_r', zmid=0, text=np.round(Mb.values, 2), texttemplate='%{text}', colorbar=dict(title='Bias (K)'))); ui.layout(fig, "Bias medio (simulato − misurato)", 380, yaxis=dict(autorange='reversed')); st.plotly_chart(fig, width='stretch')
            else:
                m = te.groupby(gr).apply(metr, include_groups=False).reset_index(); mn = no.assign(res=no.S - no.D).groupby(gr).apply(lambda d: np.sqrt((d.res ** 2).mean()), include_groups=False).rename('RMSE_nominale').reset_index() if len(no) else None
                if mn is not None: m = m.merge(mn, on=gr, how='left')
                fig = go.Figure(); fig.add_bar(x=m[gr].astype(str), y=m.RMSE, name='Calibrato', marker_color='#D9480F')
                if mn is not None: fig.add_bar(x=m[gr].astype(str), y=m.RMSE_nominale, name='Nominale', marker_color='#868E96')
                fig.add_scatter(x=m[gr].astype(str), y=te.groupby(gr).apply(lambda d: np.sqrt((d.U ** 2).mean()), include_groups=False).reindex(m[gr]).values, mode='markers', marker=dict(symbol='diamond', size=12, color='#2B8A3E'), name='Rumore di misura'); ui.layout(fig, "RMSE sul test", 380, barmode='group', yaxis_title='K', legend=dict(orientation='h', y=-0.2)); st.plotly_chart(fig, width='stretch')
                st.dataframe(m.rename(columns={'RMSE': 'RMSE (K)', 'bias': 'Bias (K)', 'MAE': 'MAE (K)', 'entro_u': '|res| ≤ u_D (%)', 'entro_2u': '|res| ≤ 2 u_D (%)', 'RMSE_nominale': 'RMSE nominale (K)'}).round(3), hide_index=True, width='stretch')
            ui.sezione("Simulato e misurato"); a, b = st.columns(2); fig = go.Figure(); colv = 'ric'; cm = dict(zip(dati.RICS, ui.COL_RIC))
            for r_ in dati.RICS:
                x = te[te.ric == r_]
                if len(x): fig.add_trace(go.Scatter(x=x.D, y=x.S, mode='markers', marker=dict(size=5, color=cm[r_], opacity=0.55), name=f"{r_} (c={dati.CONTR[r_]:.2f})"))
            lim = [float(te[['D', 'S']].min().min()), float(te[['D', 'S']].max().max())]; fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines', line=dict(color='black', dash='dash'), name='identità')); ui.layout(fig, "Simulato contro misurato (insieme di test)", 420, xaxis_title='Misurato (K)', yaxis_title='Simulato (K)', legend=dict(font=dict(size=11))); a.plotly_chart(fig, width='stretch')
            z = te.z.dropna().clip(-6, 6); xs = np.linspace(-6, 6, 200); fig = go.Figure(go.Histogram(x=z, nbinsx=40, histnorm='probability density', marker_color='#1f77b4', name='Residui standardizzati')); fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs), mode='lines', line=dict(color='#D9480F', width=3), name='Normale standard'))
            ui.layout(fig, f"Residui standardizzati (res/u_D): media {z.mean():+.2f}, scarto tipo {z.std():.2f}", 420, xaxis_title='(simulato − misurato) / u_D', legend=dict(orientation='h', y=-0.2)); b.plotly_chart(fig, width='stretch')
            st.caption("Se il modello descrivesse i dati al livello del rumore, i residui standardizzati sarebbero distribuiti come una normale standard (media 0, scarto tipo 1). Uno scarto tipo maggiore di 1 indica un errore di modello oltre il rumore.")
            if 't' in te.columns and te.t.notna().any():
                a, b = st.columns(2); g = te.groupby('t').res.agg(['mean', 'std']).reset_index(); fig = go.Figure(go.Scatter(x=g.t, y=g['mean'], mode='lines', line=dict(color='#D9480F', width=3), name='Media')); fig.add_trace(go.Scatter(x=np.r_[g.t, g.t[::-1]], y=np.r_[g['mean'] + g['std'], (g['mean'] - g['std'])[::-1]], fill='toself', fillcolor='rgba(217,72,15,0.15)', line=dict(width=0), name='± 1 scarto tipo')); fig.add_hline(y=0, line=dict(color='black', width=1)); ui.layout(fig, "Residuo in funzione del tempo", 380, xaxis_title='Tempo (s)', yaxis_title='K'); a.plotly_chart(fig, width='stretch')
                bins = pd.cut(te.D, bins=[-1, 2, 5, 10, 20, 40, 100]); gg = te.groupby(bins, observed=True).agg(rmse=('res', lambda x: np.sqrt((x ** 2).mean())), u=('U', 'mean')).reset_index(); fig = go.Figure(); fig.add_bar(x=gg.D.astype(str), y=gg.rmse, name='RMSE', marker_color='#D9480F'); fig.add_scatter(x=gg.D.astype(str), y=gg.u, mode='lines+markers', name='u_D', line=dict(color='#2B8A3E', width=3)); ui.layout(fig, "RMSE per intervallo di incremento misurato", 380, xaxis_title='ΔT misurato (K)', yaxis_title='K'); b.plotly_chart(fig, width='stretch')
        if len(te) and 'cond' in te.columns and te.cond.nunique() >= 3:
            ui.sezione("Incertezza delle metriche (bootstrap per condizione)"); rngb = np.random.default_rng(5); conds = te.cond.unique(); idxc = {c_: np.where(te.cond.values == c_)[0] for c_ in conds}; Sv, Dv = te.S.values, te.D.values; par_ = len(no) == len(te); Sn = no.S.values if par_ else None; bt = []
            for _ in range(400):
                ii = np.concatenate([idxc[c_] for c_ in rngb.choice(conds, len(conds), replace=True)]); m = mo.indici_confronto(Sv[ii], Dv[ii]); riga = [m['RMSE'], m['bias'], m['MAE'], m['R2']]
                if par_: riga.append(np.sqrt(np.mean((Sn[ii] - Dv[ii]) ** 2)) - m['RMSE'])
                bt.append(riga)
            B = np.array(bt); m0 = mo.indici_confronto(Sv, Dv); vals = [m0['RMSE'], m0['bias'], m0['MAE'], m0['R2']]; nomi_m = ['RMSE (K)', 'Bias (K)', 'MAE (K)', 'R²']
            if par_: vals.append(np.sqrt(np.mean((Sn - Dv) ** 2)) - m0['RMSE']); nomi_m.append('Riduzione di RMSE rispetto al nominale (K)')
            TB = pd.DataFrame({'Metrica': nomi_m, 'Valore': vals, 'Intervallo 95% basso': np.percentile(B, 2.5, axis=0), 'Intervallo 95% alto': np.percentile(B, 97.5, axis=0)}); a, b = st.columns([1.1, 1]); a.dataframe(TB.round(3), hide_index=True, width='stretch')
            fig = go.Figure(go.Histogram(x=B[:, 0], nbinsx=30, marker_color='#D9480F')); fig.add_vline(x=m0['RMSE'], line=dict(color='black', dash='dash')); ui.layout(fig, f"RMSE: distribuzione bootstrap ({te.cond.nunique()} condizioni)", 300, xaxis_title='K', showlegend=False); b.plotly_chart(fig, width='stretch'); st.caption("Le condizioni (e non i singoli punti) sono ricampionate con reinserimento, perché i punti di una stessa condizione sono correlati: l'intervallo riflette quanto la metrica dipende dalle condizioni scelte per il test.")
        TH = pd.DataFrame([dict(Risultato=i + 1, Modello=x['r'].get('cand'), Fold=x['r'].get('fold', '—'), **{k: v for k, v in x['r'].get('theta', {}).items()}, **{f'sd_{k}': v for k, v in x['r'].get('sd', {}).items()}) for i, x in zip(idx, R)])
        if len(TH) > 1:
            ui.sezione("Stabilità dei parametri fra i risultati"); st.dataframe(TH.round(3), hide_index=True, width='stretch'); pc = [c for c in TH.columns if c in cp.PRIOR]; fig = go.Figure()
            for p in pc: fig.add_trace(go.Scatter(x=TH['Risultato'].astype(str) + ' · ' + TH['Modello'].astype(str), y=TH[p], mode='markers+lines', error_y=dict(type='data', array=1.96 * TH.get(f'sd_{p}', pd.Series(0, index=TH.index)).fillna(0)), name=p))
            ui.layout(fig, "Parametri stimati (intervallo 95%)", 380, yaxis_title='Moltiplicatore'); st.plotly_chart(fig, width='stretch')
        ui.callout("Se il divario fra RMSE di test e di addestramento è piccolo e l'RMSE di test si avvicina al rumore di misura, il modello generalizza; un divario grande indica sovradattamento. La validazione è una stima dell'accuratezza del modello e non un esame da superare.")
ui.piede()
