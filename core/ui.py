"""Tema, componenti grafici e utilita' comuni dell'interfaccia."""
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

COL = dict(strutt='#1f77b4', conf='#ff7f0e', rosso='#C92A2A', verde='#2B8A3E', giallo='#E8B100', grigio='#868E96', blu_scuro='#0B3954', teal='#0F766E', caldo='#D9480F')
COL_RIC = ['#2166ac', '#67a9cf', '#bdbdbd', '#ef8a62', '#b2182b']          # dal contrasto piu' basso (B) al piu' alto (C)
COL_PAR = {'sig_m20': '#1f77b4', 'sig_p20': '#17becf', 'fc_m': '#d62728', 'fc_p': '#ff9896', 'alfa_m': '#2ca02c', 'alfa_p': '#98df8a', 'fk_m': '#9467bd', 'fk_p': '#c5b0d5', 'fh': '#7f7f7f'}
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

pio.templates['ohmico'] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, size=13, color='#16212E'), paper_bgcolor='white', plot_bgcolor='white', colorway=[COL['strutt'], COL['conf'], '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf'],
    xaxis=dict(showline=True, linecolor='#495057', mirror=True, gridcolor='#E9ECEF', zeroline=False, ticks='outside'), yaxis=dict(showline=True, linecolor='#495057', mirror=True, gridcolor='#E9ECEF', zeroline=False, ticks='outside'),
    legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#DEE2E6', borderwidth=1), margin=dict(l=60, r=20, t=50, b=50), hoverlabel=dict(font_family=FONT)))
pio.templates.default = 'ohmico'

CSS = """
<style>
html, body, [class*="css"] { font-family: Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B3954 0%, #123B5C 55%, #1B4F72 100%); }
[data-testid="stSidebar"] * { color: #EAF2F8 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a { border-radius: 8px; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,0.10); }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * { color: #16212E !important; }
.hero { background: linear-gradient(120deg, #0B3954 0%, #145374 55%, #D9480F 130%); color: white; padding: 26px 32px; border-radius: 16px; margin-bottom: 18px; box-shadow: 0 6px 24px rgba(11,57,84,0.25); }
.hero h1 { color: white; font-size: 2.0rem; margin: 0 0 6px 0; padding: 0; line-height: 1.15; }
.hero p { color: #DCEBF5; font-size: 1.02rem; margin: 0; max-width: 1100px; }
.hero .tag { display: inline-block; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28); padding: 2px 11px; border-radius: 999px; font-size: 0.78rem; margin: 10px 8px 0 0; }
.card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px; height: 100%; }
.card h4 { margin: 0 0 6px 0; color: #0B3954; font-size: 1.02rem; }
.card p { margin: 0; color: #334155; font-size: 0.93rem; line-height: 1.45; }
.kpi { background: white; border: 1px solid #E2E8F0; border-left: 5px solid #D9480F; border-radius: 12px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(15,23,42,0.06); }
.kpi .v { font-size: 1.75rem; font-weight: 700; color: #0B3954; line-height: 1.1; }
.kpi .l { font-size: 0.82rem; color: #475569; margin-top: 2px; }
.kpi .s { font-size: 0.75rem; color: #64748B; margin-top: 2px; }
.callout { border-left: 5px solid #0F766E; background: #F0FDFA; padding: 10px 16px; border-radius: 0 10px 10px 0; margin: 8px 0 14px 0; color: #134E4A; font-size: 0.93rem; }
.callout.warn { border-left-color: #E8B100; background: #FFFBEB; color: #713F12; }
.callout.risk { border-left-color: #C92A2A; background: #FFF5F5; color: #7F1D1D; }
.step { display: flex; align-items: center; gap: 12px; padding: 8px 0; }
.step .n { background: #0B3954; color: white; border-radius: 50%; min-width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; }
.step .t { font-size: 0.93rem; color: #1E293B; }
.badge { display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 0.74rem; font-weight: 600; margin-right: 6px; }
.badge.ok { background: #D3F9D8; color: #1B5E20; } .badge.mid { background: #FFF3BF; color: #7A5B00; } .badge.no { background: #FFE3E3; color: #8A1C1C; } .badge.nd { background: #E9ECEF; color: #495057; }
.sezione { color: #0B3954; border-bottom: 2px solid #D9480F; padding-bottom: 4px; margin: 22px 0 10px 0; font-size: 1.25rem; font-weight: 700; }
div[data-testid="stMetric"] { background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(15,23,42,0.05); }
button[kind="primary"] { border-radius: 10px; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { background: #F1F5F9; border-radius: 8px 8px 0 0; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background: #0B3954 !important; color: white !important; }
.stTabs [aria-selected="true"] p { color: white !important; }
.footer { color: #64748B; font-size: 0.8rem; text-align: center; margin-top: 30px; border-top: 1px solid #E2E8F0; padding-top: 10px; }
</style>
"""


def applica_tema(): st.markdown(CSS, unsafe_allow_html=True)


def hero(titolo, sottotitolo, tags=()):
    t = ''.join(f'<span class="tag">{x}</span>' for x in tags); st.markdown(f'<div class="hero"><h1>{titolo}</h1><p>{sottotitolo}</p>{t}</div>', unsafe_allow_html=True)


def kpi(col, valore, etichetta, sotto=''):
    col.markdown(f'<div class="kpi"><div class="v">{valore}</div><div class="l">{etichetta}</div><div class="s">{sotto}</div></div>', unsafe_allow_html=True)


def card(col, titolo, testo): col.markdown(f'<div class="card"><h4>{titolo}</h4><p>{testo}</p></div>', unsafe_allow_html=True)


def callout(testo, tipo=''): st.markdown(f'<div class="callout {tipo}">{testo}</div>', unsafe_allow_html=True)


def sezione(t): st.markdown(f'<div class="sezione">{t}</div>', unsafe_allow_html=True)


def passo(n, testo): st.markdown(f'<div class="step"><div class="n">{n}</div><div class="t">{testo}</div></div>', unsafe_allow_html=True)


def badge(testo, tipo='nd'): return f'<span class="badge {tipo}">{testo}</span>'


def piede(): st.markdown('<div class="footer">Gemello digitale del riscaldamento ohmico multifase: cella 99 × 54 × 66 mm · motore FEM 3D (scikit-fem) · risultati del protocollo di verifica, sensitività, calibrazione e validazione</div>', unsafe_allow_html=True)


def layout(fig, titolo=None, h=460, **kw):
    fig.update_layout(title=dict(text=titolo, x=0.01, font=dict(size=15)) if titolo else None, height=h, **kw); return fig
