"""Simulatore 3D accoppiato della cella: impostazioni, curve alle sonde, campi, potenza ed energia, fasi, mesh e confronto."""
import time, json
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import ui, dati, calibra
from core import motore as mo

ui.applica_tema()
ui.hero("Simulatore del gemello digitale", "Simulazione elettro-termica 3D accoppiata di un caso della cella: curve alle sonde, campi di temperatura e di conducibilità, ripartizione della potenza fra le fasi e bilancio energetico, con il confronto diretto con una prova sperimentale.", ["FEM P1 · Eulero implicito", "σ(T) e proprietà dipendenti dalla temperatura", "scambio superficiale h(T)"])
S = st.session_state; S.setdefault('sim', None); S.setdefault('confronto', [])
cc, cr = st.columns([1, 3.1], gap="large")
with cc:
    with st.container(border=True):
        st.markdown("##### 1 · Caso")
        ric = st.selectbox("Ricetta", sorted(dati.AB), format_func=lambda r: f"{r} · sale {dati.SALE[r]:g}% · c = {dati.CONTR[r]:.2f}", index=2, key="sm_ric")
        conf = st.selectbox("Configurazione", dati.CFG, format_func=lambda c: f"{c} ({dati.NPOLP[c]} polpette)", key="sm_conf")
        vsel = st.radio("Tensione efficace", ["30 V", "40 V", "50 V", "altra"], horizontal=True, index=1, key="sm_vsel")
        V = st.number_input("Tensione (V)", 5.0, 200.0, 40.0, 1.0, key="sm_V") if vsel == "altra" else float(vsel.split()[0])
        provs = calibra.catalogo_prove(); prov = provs[(provs.ric == ric) & (provs.V == int(round(V))) & (provs.cfg == conf)]
        usa_prova = False; run_sel = None
        if len(prov):
            usa_prova = st.checkbox(f"Confronta con una prova sperimentale ({len(prov)} disponibili)", value=True, key="sm_usa")
            if usa_prova: run_sel = st.selectbox("Prova", list(prov.run), format_func=lambda r: f"prova {r} · T0 purea {prov[prov.run == r].T0_purea.iloc[0]:.1f} °C", key="sm_run")
        if usa_prova:
            rr = calibra.prova(run_sel); T0, T0s = rr['T0m'], rr['T0s']; st.caption(f"Condizioni iniziali dalla prova: purea {T0:.1f} °C, polpette {', '.join(f'{x:.1f}' for x in T0s)} °C.")
        else:
            T0 = st.number_input("Temperatura iniziale della purea (°C)", 0.0, 90.0, 20.0, 0.5, key="sm_T0"); T0p = st.number_input("Temperatura iniziale delle polpette (°C)", 0.0, 90.0, float(T0), 0.5, key="sm_T0p"); T0s = [T0p] * dati.NPOLP[conf]
        T_amb = st.number_input("Temperatura dell'aria (°C)", 0.0, 50.0, 21.0, 0.5, key="sm_Ta")
    with st.container(border=True):
        st.markdown("##### 2 · Discretizzazione")
        fams = ['strutturata'] + (['conforme'] if mo.GMSH else []); fam = st.radio("Famiglia di mesh", fams, format_func=lambda f: mo.FAMIGLIE[f], key="sm_fam")
        if not mo.GMSH: st.caption("La mesh conforme richiede gmsh (non disponibile qui).")
        liv = st.select_slider("Livello di mesh", options=[0, 1, 2, 3], value=1, format_func=lambda k: f"{k} · {mo.NODI_LIVELLO[k]:,} nodi".replace(',', ' '), key="sm_liv")
        dt = st.select_slider("Passo temporale (s)", options=[5.0, 10.0, 20.0], value=10.0, key="sm_dt"); dur = st.number_input("Durata (s)", 20.0, 900.0, 270.0, 10.0, key="sm_dur")
        stima = mo.stima_costo(conf, liv) * dur / 270 * (10.0 / dt); st.caption(f"Tempo di calcolo stimato: circa **{stima:.0f} s**")
    with st.expander("3 · Parametri e modelli", expanded=False):
        nom, caso0 = mo.nominali(ric, V, conf); mult = {}
        st.caption("Moltiplicatori dei nove parametri (1 = valore nominale)")
        for par in dati.PAR:
            mult[par] = st.slider(f"{dati.PAR_TXT[par]} · {dati.PAR_DESC[par]}", 0.5, 1.5, 1.0, 0.01, key=f"sm_m_{par}", help="Valore nominale: " + (f"{nom[par]:.4g}" if par in ('sig_m20', 'alfa_m', 'sig_p20', 'alfa_p') else "1"))
        prop = st.selectbox("Proprietà termofisiche", ['choi_okos_T', 'choi_okos_Tref', 'manuali'], format_func=lambda x: {'choi_okos_T': 'Choi–Okos dipendenti da T', 'choi_okos_Tref': 'Choi–Okos a T di riferimento', 'manuali': 'Valori manuali costanti'}[x], key="sm_prop")
        hm = st.selectbox("Scambio sulla superficie libera", ['naturale_irraggiamento', 'costante', 'naturale_irraggiamento_evaporazione'], format_func=lambda x: {'naturale_irraggiamento': 'Convezione naturale + irraggiamento', 'costante': 'h costante', 'naturale_irraggiamento_evaporazione': '+ evaporazione'}[x], key="sm_hm")
        hc = st.number_input("h costante (W/m²/K)", 1.0, 50.0, 10.0, 0.5, key="sm_hc", disabled=(hm != 'costante'))
    avvia = st.button("▶ Avvia la simulazione", type="primary", width='stretch', key="sm_go")
    if avvia:
        barra = st.progress(0.0, text="Simulazione in corso…"); ass = dict(prop_modo=prop, h_modo=hm, h_conv=hc)
        try:
            ris = mo.simula(ric, V, conf, fam, liv, T0=T0, T0_sfere=T0s, T_amb=T_amb, dt=dt, durata=dur, mult=mult, assoluti=ass, snap_n=13, avanzamento=lambda x: barra.progress(min(1.0, x), text=f"Simulazione in corso… {100 * x:.0f}%"))
            S.sim = dict(ris=ris, ric=ric, V=V, conf=conf, fam=fam, liv=liv, dt=dt, dur=dur, T0=T0, T0s=T0s, T_amb=T_amb, mult=mult, prop=prop, hm=hm, run=run_sel if usa_prova else None, label=f"{ric} · {V:g} V · {conf} · {'conf.' if fam == 'conforme' else 'strutt.'} L{liv}")
            barra.empty()
        except Exception as e:
            barra.empty(); st.error(f"Simulazione non riuscita: {e}")

with cr:
    sim = S.sim
    if sim is None:
        ui.callout("Imposta il caso a sinistra e premi <b>Avvia la simulazione</b>. Il livello 1 (circa 2 300 nodi) completa 270 s in pochi secondi; i livelli più fini sono più accurati e più lenti.")
    else:
        ris = sim['ris']; se = ris['serie']; info = ris['info']; ctx = ris['ctx']; t = se['t'].values; fase = ['polpetta' if dati.OCC[sim['conf']][i] else 'purea' for i in range(3)]
        n_nodi_t = f"{info['n_nodi']:,}".replace(',', ' '); n_el_t = f"{info['n_el']:,}".replace(',', ' '); t_calc = f"{ris['tempo']:.1f}"
        st.markdown("**Risultati · " + sim['label'] + "** &nbsp; " + ui.badge(n_nodi_t + " nodi", 'nd') + ui.badge(n_el_t + " tetraedri", 'nd') + ui.badge("calcolo " + t_calc + " s", 'nd'), unsafe_allow_html=True)
        k = st.columns(6)
        ui.kpi(k[0], f"{se.T_P1.iloc[-1]:.1f} °C", "T alla sonda P1", f"ΔT = {se.T_P1.iloc[-1] - se.T_P1.iloc[0]:.1f} K"); ui.kpi(k[1], f"{se.T_max.iloc[-1]:.1f} °C", "T massima", f"nodo più caldo"); ui.kpi(k[2], f"{se.P.iloc[-1]:.0f} W", "Potenza finale", f"iniziale {se.P.iloc[0]:.0f} W")
        ui.kpi(k[3], f"{100 * se.frac_P_p.iloc[-1]:.0f}%", "Potenza nelle polpette", f"volume 12%"); ui.kpi(k[4], f"{se.E_in.iloc[-1] / 1000:.1f} kJ", "Energia fornita", f"perdite {100 * se.E_loss.iloc[-1] / max(se.E_in.iloc[-1], 1):.1f}%"); ui.kpi(k[5], f"{100 * abs(se.chiusura_energia.iloc[-1]):.3f}%", "Chiusura energetica", "residuo del bilancio")
        tabs = st.tabs(["Sonde", "Campi 3D", "Potenza ed energia", "Fasi e punti critici", "Mesh e modello", "Confronto", "Esporta"])
        with tabs[0]:
            modo = st.radio("Mostra", ["Temperatura (°C)", "Incremento ΔT (K)"], horizontal=True, key="sm_modo"); fig = go.Figure(); cols = ['#1f77b4', '#d62728', '#2ca02c']
            for i in range(3):
                y = se[f'T_P{i + 1}'].values; y = y if modo.startswith('Temp') else y - y[0]; fig.add_trace(go.Scatter(x=t, y=y, mode='lines', line=dict(color=cols[i], width=3), name=f"P{i + 1} · modello ({fase[i]})"))
            if sim['run'] is not None:
                rr = calibra.prova(sim['run']); sf = [j for j in range(3) if dati.OCC[sim['conf']][j]]
                for i in range(3):
                    base = rr['T0s'][sf.index(i)] if i in sf else rr['T0m']; yy = rr['dT'][:, i] + (base if modo.startswith('Temp') else 0); fig.add_trace(go.Scatter(x=rr['t'], y=yy, mode='markers', marker=dict(color=cols[i], size=5, symbol='circle-open'), name=f"P{i + 1} · misura"))
            ui.layout(fig, None, 470, xaxis_title='Tempo (s)', yaxis_title=modo, legend=dict(orientation='h', y=-0.2)); st.plotly_chart(fig, width='stretch')
            if sim['run'] is not None:
                rr = calibra.prova(sim['run']); n = len(rr['t']); idx = np.arange(2, n); rows = []; sf = [j for j in range(3) if dati.OCC[sim['conf']][j]]
                for i in range(3):
                    simi = np.interp(rr['t'][idx], t, se[f'T_P{i + 1}'].values - se[f'T_P{i + 1}'].values[0]); m = mo.indici_confronto(simi, rr['dT'][idx, i]); rows.append({'Sonda': f'P{i + 1}', 'Fase': fase[i], 'RMSE (K)': m['RMSE'], 'Bias medio (K)': m['bias'], 'MAE (K)': m['MAE'], 'R²': m['R2'], 'ΔT finale misura (K)': rr['dT'][-1, i], 'ΔT finale modello (K)': float(np.interp(rr['t'][-1], t, se[f'T_P{i + 1}'].values - se[f'T_P{i + 1}'].values[0]))})
                st.markdown("**Confronto con la prova (incrementi, dai 10 s)**"); tab = pd.DataFrame(rows); st.dataframe(tab.round(3), hide_index=True, width='stretch')
                ud = calibra.uD_stima(1.5); ref = float(calibra.uD(np.array([np.nanmean(np.abs(rr['dT'][-1]))]), ud)[0]); st.caption(f"Incertezza di misura dell'incremento finale medio: circa {ref:.2f} K (modello del rumore, paragrafo «Cella, ricette e dati»). Un modello nominale non calibrato può discostarsi di più.")
        with tabs[1]:
            snaps = info['snap']
            if not snaps: st.info("Nessuna istantanea disponibile.")
            else:
                c1, c2, c3 = st.columns(3); campo = c1.selectbox("Campo", ["Temperatura (°C)", "Conducibilità σ (S/m)", "Fase φ (polpetta = 1)"], key="sm_campo"); piano = c2.radio("Piano di taglio", ["xy (dall'alto)", "xz (frontale)", "yz (trasversale)"], key="sm_piano")
                L = dict(x=ctx.geo['Lx'] * 1000, y=ctx.geo['Ly'] * 1000, z=ctx.geo['Lz'] * 1000); asse = {'x': 'z', 'f': 'y', 'y': 'x'}[{'xy': 'x', 'xz': 'f', 'yz': 'y'}[piano.split()[0]]]; pos = c3.slider(f"Posizione lungo {asse} (mm)", 0.0, L[asse], L[asse] / 2, 0.5, key="sm_pos")
                ti = st.select_slider("Istante", options=list(range(len(snaps))), value=len(snaps) - 1, format_func=lambda i: f"{snaps[i][0]:.0f} s", key="sm_ti")
                def campo_val(T):
                    if campo.startswith('Temp'): return np.asarray(T)
                    if campo.startswith('Cond'): return mo.sigma_nodale(ctx, T, ris['p'])
                    return np.asarray(ctx.phi_n)
                Zs = [mo.fetta(ctx, campo_val(Tt), asse, pos)[2] for _, Tt in snaps]; zmin, zmax = float(np.nanmin([np.nanmin(z) for z in Zs])), float(np.nanmax([np.nanmax(z) for z in Zs]))
                u, v, Z = mo.fetta(ctx, campo_val(snaps[ti][1]), asse, pos); _, _, Phi = mo.fetta(ctx, ctx.phi_n, asse, pos); et = dict(x=('y', 'z'), y=('x', 'z'), z=('x', 'y'))[asse]
                fig = go.Figure(go.Heatmap(x=u, y=v, z=Z, colorscale='Inferno' if campo.startswith('Temp') else ('Viridis' if campo.startswith('Cond') else 'Greys'), zmin=zmin, zmax=zmax, colorbar=dict(title=campo.split()[0]), hovertemplate=f'{et[0]}=%{{x:.1f}} mm<br>{et[1]}=%{{y:.1f}} mm<br>valore=%{{z:.3f}}<extra></extra>'))
                fig.add_trace(go.Contour(x=u, y=v, z=Phi, contours=dict(start=0.5, end=0.5, size=1, coloring='none'), line=dict(color='white', width=2), showscale=False, hoverinfo='skip', name='interfaccia'))
                ui.layout(fig, f"{campo} · piano {asse} = {pos:.1f} mm · t = {snaps[ti][0]:.0f} s", 520, xaxis_title=f"{et[0]} (mm)", yaxis=dict(title=f"{et[1]} (mm)", scaleanchor='x')); st.plotly_chart(fig, width='stretch')
                st.caption("La linea bianca è l'interfaccia fra purea e polpette (φ = 0.5). La scala di colore è fissa su tutti gli istanti, per vedere l'evoluzione.")
        with tabs[2]:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Potenza elettrica", "Ripartizione della potenza"), horizontal_spacing=0.10)
            fig.add_trace(go.Scatter(x=t, y=se.P, name='Totale', line=dict(color='#0B3954', width=3)), row=1, col=1); fig.add_trace(go.Scatter(x=t, y=se.P_p, name='Polpette', line=dict(color='#E8590C', width=2.5)), row=1, col=1); fig.add_trace(go.Scatter(x=t, y=se.P_m, name='Purea', line=dict(color='#2B8A3E', width=2.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=100 * se.frac_P_p, name='Π_p (%)', line=dict(color='#E8590C', width=3), showlegend=False), row=1, col=2); fig.add_hline(y=12.0, line=dict(color='#868E96', dash='dot'), annotation_text='frazione di volume 12%', row=1, col=2)
            fig.update_xaxes(title='Tempo (s)'); fig.update_yaxes(title='W', row=1, col=1); fig.update_yaxes(title='Potenza nelle polpette (%)', row=1, col=2); fig.update_layout(height=410, legend=dict(orientation='h', y=-0.22)); st.plotly_chart(fig, width='stretch')
            fig = go.Figure()
            for col, nm, c in (('E_in', 'Energia fornita', '#0B3954'), ('E_stored', 'Energia immagazzinata', '#E8590C'), ('E_loss', 'Perdite superficiali', '#868E96')): fig.add_trace(go.Scatter(x=t, y=se[col] / 1000, name=nm, line=dict(width=3, color=c)))
            fig.add_trace(go.Scatter(x=t, y=se.E_stored_p / 1000, name='  di cui polpette', line=dict(width=2, color='#F08C00', dash='dash'))); fig.add_trace(go.Scatter(x=t, y=se.E_stored_m / 1000, name='  di cui purea', line=dict(width=2, color='#2B8A3E', dash='dash')))
            ui.layout(fig, "Bilancio energetico", 400, xaxis_title='Tempo (s)', yaxis_title='kJ', legend=dict(orientation='h', y=-0.25)); st.plotly_chart(fig, width='stretch')
            st.caption(f"Chiusura del bilancio (E_in − E_imm. − E_perdite)/E_in: {100 * se.chiusura_energia.iloc[-1]:.2e}%. Sbilancio massimo delle correnti: {se.sbilancio_I.abs().max():.2e}; P − V·I: {se.P_meno_VI.abs().max():.2e} W.")
        with tabs[3]:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Temperature delle fasi", "Differenza polpette − purea"), horizontal_spacing=0.10)
            for col, nm, c in (('T_media_polp', 'Media polpette', '#E8590C'), ('T_media_pur', 'Media purea', '#2B8A3E'), ('T_max', 'Massima', '#C92A2A'), ('T_min', 'Minima', '#1971C2'), ('T_polp_min_nodo', 'Minima nelle polpette', '#862E9C')): fig.add_trace(go.Scatter(x=t, y=se[col], name=nm, line=dict(width=2.6, color=c)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=se.T_media_polp - se.T_media_pur, name='T̄_p − T̄_m', line=dict(width=3, color='#0B3954'), showlegend=False), row=1, col=2); fig.add_hline(y=0, line=dict(color='#868E96', dash='dot'), row=1, col=2)
            fig.update_xaxes(title='Tempo (s)'); fig.update_yaxes(title='°C', row=1, col=1); fig.update_yaxes(title='K', row=1, col=2); fig.update_layout(height=430, legend=dict(orientation='h', y=-0.22)); st.plotly_chart(fig, width='stretch')
            ths = [40, 50, 60, 70, 80, 90]; rows = []
            for nm, col in (('P1', 'T_P1'), ('P2', 'T_P2'), ('P3', 'T_P3'), ('Massima', 'T_max'), ('Media polpette', 'T_media_polp'), ('Media purea', 'T_media_pur')):
                r = {'Punto': nm}
                for th in ths: idx = np.where(se[col].values >= th)[0]; r[f'{th} °C'] = f"{t[idx[0]]:.0f} s" if len(idx) else '—'
                rows.append(r)
            st.markdown("**Tempo per raggiungere una temperatura**"); st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            vel = np.gradient(se.T_media_pur.values, t); st.caption(f"Velocità media di riscaldamento della purea: {np.mean(vel[1:]):.3f} K/s (massima {vel.max():.3f} K/s).")
        with tabs[4]:
            m1, m2, m3, m4 = st.columns(4); m1.metric("Nodi", f"{info['n_nodi']:,}".replace(',', ' ')); m2.metric("Tetraedri", f"{info['n_el']:,}".replace(',', ' ')); m3.metric("Passo effettivo h_eff", f"{info['h_eff'] * 1000:.2f} mm"); m4.metric("Elementi su un diametro", f"{ctx.geo['D'] / info['h_eff']:.1f}")
            ev = info.get('volume_phi'); ee = info.get('volume_esatto')
            if ev and ee: st.caption(f"Volume delle polpette: {ev * 1e6:.2f} cm³ nel modello, {ee * 1e6:.2f} cm³ esatto (differenza {100 * (ev - ee) / ee:+.2f}%).")
            a, b = st.columns(2)
            with a:
                u, v, Phi = mo.fetta(ctx, ctx.phi_n, 'z', ctx.geo['Lz'] * 500, 220, 120); fig = go.Figure(go.Heatmap(x=u, y=v, z=Phi, colorscale='OrRd', zmin=0, zmax=1, colorbar=dict(title='φ')))
                ui.layout(fig, "Campo di fase φ (piano a metà altezza)", 330, xaxis_title='x (mm)', yaxis=dict(title='y (mm)', scaleanchor='x')); st.plotly_chart(fig, width='stretch')
            with b:
                Ts = np.linspace(10, 95, 100); heq = mo.me.h_equivalente(Ts, ris['p'], ctx.L_car); fig = go.Figure(go.Scatter(x=Ts, y=heq, line=dict(color='#0B3954', width=3)))
                ui.layout(fig, "Coefficiente di scambio equivalente h(T)", 330, xaxis_title='Temperatura della superficie (°C)', yaxis_title='h (W/m²/K)'); st.plotly_chart(fig, width='stretch')
            T = np.linspace(10, 90, 100); a_m, b_m = ris['p']['matrice_a'], ris['p']['matrice_b']; a_p, b_p = ris['p']['polpetta_a'], ris['p']['polpetta_b']
            fig = go.Figure(); fig.add_trace(go.Scatter(x=T, y=a_m * T + b_m, name='Purea', line=dict(width=3, color='#2B8A3E'))); fig.add_trace(go.Scatter(x=T, y=a_p * T + b_p, name='Polpette', line=dict(width=3, color='#E8590C')))
            ui.layout(fig, "Conducibilità elettrica usata nel modello", 320, xaxis_title='Temperatura (°C)', yaxis_title='σ (S/m)'); st.plotly_chart(fig, width='stretch')
        with tabs[5]:
            c1, c2 = st.columns([1, 3])
            if c1.button("Aggiungi questa simulazione al confronto", key="sm_add"): S.confronto.append(dict(label=sim['label'], serie=se.copy())); st.rerun()
            if c1.button("Svuota il confronto", key="sm_clr"): S.confronto = []; st.rerun()
            if not S.confronto: c2.info("Salva una o più simulazioni per confrontarle (per esempio al variare di tensione, ricetta o parametri).")
            else:
                grand = c2.selectbox("Grandezza", ['T_P1', 'T_P2', 'T_P3', 'T_max', 'T_media_polp', 'T_media_pur', 'P', 'frac_P_p', 'E_in'], format_func=lambda x: dati.Q[x][1] if x in dati.Q else x, key="sm_gr"); fig = go.Figure()
                for it in S.confronto: fig.add_trace(go.Scatter(x=it['serie'].t, y=it['serie'][grand], name=it['label'], line=dict(width=2.8)))
                ui.layout(fig, None, 440, xaxis_title='Tempo (s)', yaxis_title=dati.Q[grand][2] if grand in dati.Q else '', legend=dict(orientation='h', y=-0.22)); st.plotly_chart(fig, width='stretch')
        with tabs[6]:
            st.download_button("Scarica le serie temporali (CSV)", se.to_csv(index=False), "simulazione_serie.csv", "text/csv", key="sm_dl")
            st.download_button("Scarica le impostazioni (JSON)", json.dumps({k: v for k, v in sim.items() if k not in ('ris',)}, default=str, indent=1), "simulazione_impostazioni.json", "application/json", key="sm_dlj")
            st.dataframe(se.round(4), hide_index=True, width='stretch', height=300)
ui.piede()
