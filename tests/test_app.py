"""Collaudo dell'app con il framework di test di Streamlit.

Esecuzione:   python -m pytest tests -q        oppure      python tests/test_app.py
Le prove con simulazioni usano la mesh più grossolana e durano pochi secondi ciascuna.
"""
import os, sys, time
AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, AQUI)
from streamlit.testing.v1 import AppTest

PAGINE = ['panoramica', 'cella_dati', 'simulatore', 'nominale', 'verifica', 'sensitivita', 'dossier', 'calibrazione', 'validazione', 'scenari', 'modello_base', 'guida']


def apri(pagina, timeout=250):
    at = AppTest.from_file(os.path.join(AQUI, 'streamlit_app.py'), default_timeout=timeout); at.run(); at.switch_page(f'pagine/{pagina}.py'); at.run(); return at


def senza_errori(at): return [e.value for e in at.exception] == [] and [e.value for e in at.error] == []


def test_tutte_le_pagine_si_aprono():
    for p in PAGINE: assert senza_errori(apri(p)), f"errore nella pagina {p}"


def test_dossier_cambia_caso():
    at = apri('dossier')
    for r in ('B', 'C'): at.selectbox(key='do_r').select(r).run(); assert senza_errori(at)


def test_modello_nominale_contro_misure():
    at = apri('nominale'); at.slider(key='no_n').set_value(6).run(); at.button(key='no_go').click().run(); assert senza_errori(at) and at.session_state['nominale']['P'].shape[0] > 100


def test_scenari_ottimizzazione_incertezza():
    at = apri('scenari'); at.button(key='sc_go').click().run(); assert senza_errori(at)
    at.button(key='op_go').click().run(); assert senza_errori(at)
    at.slider(key='un_n').set_value(8).run(); at.button(key='un_go').click().run(); assert senza_errori(at) and len(at.session_state['ens']['M']) == 8


def test_calibrazione_ridotta_e_validazione():
    at = apri('calibrazione'); at.multiselect(key='ca_r').set_value(['E']).run(); at.multiselect(key='ca_c').set_value(['Tris', 'Middle']).run()
    at.slider(key='ca_ntr').set_value(4).run(); at.slider(key='ca_nte').set_value(3).run(); at.slider(key='ca_nfev').set_value(2).run(); at.button(key='ca_go').click().run(); assert senza_errori(at)
    S = at.session_state; assert S['cal'] is not None and S['cal']['r'].get('cov')
    at2 = AppTest.from_file(os.path.join(AQUI, 'streamlit_app.py'), default_timeout=250); at2.session_state['risultati'] = S['risultati']; at2.session_state['cal'] = S['cal']; at2.run(); at2.switch_page('pagine/validazione.py'); at2.run(); assert senza_errori(at2)


if __name__ == '__main__':
    for nome, f in sorted((k, v) for k, v in globals().items() if k.startswith('test_')):
        t0 = time.time(); f(); print(f"ok  {nome:46s} {time.time() - t0:6.1f} s")
