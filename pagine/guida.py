"""Guida: modello, studi e criteri, uso dell'app, glossario e riferimenti."""
import streamlit as st
from core import ui, dati

ui.applica_tema()
ui.hero("Guida e metodi", "Il modello matematico, i criteri degli studi numerici, come usare l'app e quali analisi vanno eseguite fuori dall'app, con il glossario e i riferimenti.", ["modello", "criteri", "uso", "riferimenti"])
t1, t2, t3, t4, t5 = st.tabs(["Il modello", "Studi e criteri", "Come usare l'app", "Glossario", "Riferimenti"])
with t1:
    st.markdown("Il gemello digitale risolve in 3D, agli elementi finiti lineari (P1) e con Eulero implicito, il problema elettrico quasi statico accoppiato al bilancio di energia. Le grandezze elettriche sono valori efficaci.")
    st.markdown("**Problema elettrico** (potenziale efficace ψ, elettrodi alle facce x = 0 e x = L)"); st.latex(r"\nabla\cdot\big(\sigma(T,\mathbf{x})\,\nabla\psi\big)=0,\qquad \psi=V_{rms}\ \text{su un elettrodo},\ \ \psi=0\ \text{sull'altro}")
    st.markdown("**Sorgente di Joule e bilancio di energia**"); st.latex(r"q=\sigma\,\lvert\nabla\psi\rvert^{2},\qquad \rho c_p\,\frac{\partial T}{\partial t}=\nabla\cdot(k\,\nabla T)+q")
    st.markdown("**Conducibilità elettrica di ciascuna fase**, lineare nella temperatura (con i coefficienti delle ricette), e media pesata dalla frazione di fase φ:"); st.latex(r"\sigma_i(T)=a_i\,T+b_i,\qquad \sigma=(1-\varphi)\,\sigma_m+\varphi\,\sigma_p")
    st.markdown("**Superficie libera** (convezione naturale e irraggiamento, con coefficiente di scambio equivalente h(T)):"); st.latex(r"-k\,\frac{\partial T}{\partial n}=h(T)\,(T-T_{amb})")
    st.markdown("Le proprietà termofisiche (conducibilità, densità e calore specifico) seguono le correlazioni di Choi e Okos in funzione della composizione e della temperatura. Le pareti laterali sono isolanti, e gli elettrodi coprono interamente le facce estreme.")
    c = st.columns(2)
    ui.card(c[0], "Griglie strutturate", "L'interfaccia fra purea e polpette è rappresentata da una transizione diffusa della frazione di fase, la cui larghezza è proporzionale alla dimensione della mesh. Sono semplici e adatte a reti neurali convolutive, ma convergono più lentamente sulla ripartizione della potenza.")
    ui.card(c[1], "Mesh conformi", "Le mesh tetraedriche seguono l'interfaccia delle sfere (interfaccia netta, generate con gmsh). A pari nodi sono più accurate sulla ripartizione della potenza e convergono più regolarmente, a un costo di calcolo maggiore.")
with t2:
    st.markdown("**Incertezza numerica relativa** (in percentuale della scala della grandezza: per le temperature, l'incremento del nodo più caldo):"); st.latex(r"\tilde U_{num}=100\,\frac{U_{num}}{s}")
    st.markdown(r"Si stima con il GCI sulle tre mesh più fini e con un adattamento ai minimi quadrati di $f=f_0+a\,h^p$ su tutti i livelli. Tre esiti: **dimostrata** (regime monotono e incertezza sotto soglia), **non dimostrata ma sotto soglia** (regime non monotono, incertezza sotto soglia con il ripiego conservativo) e **sopra soglia**.")
    st.markdown("**Soglie.** Per le temperature, misurate, si applica la regola della terza parte: l'incertezza numerica non supera un terzo di quella di misura,"); st.latex(r"U_{num}\le\frac{1}{3}\,u_D(\Delta T),\qquad u_D(\Delta T)=\sqrt{(a\,\Delta T)^2+b^2}")
    st.markdown("con $a\\approx0.07$ e $b\\approx0.32$ K stimati dalle ripetizioni: ne risulta una soglia di circa 2.5% della scala, applicata anche alle grandezze che alimentano le temperature. Per gli indicatori locali non misurati e per $T_{\\max}$ la soglia è del 5%, una scelta motivata e non derivata da una misura.")
    st.markdown("**Quota di sensitività locale** (effetto $e_j$ di +10% del parametro $j$ sulle temperature alle sonde):"); st.latex(r"q_j=100\,\frac{e_j}{\sum_k e_k}")
    st.markdown("**Calibrazione.** Con $\\theta$ i moltiplicatori dei parametri, si minimizza la somma dei quadrati dei residui degli incrementi di temperatura pesati con l'incertezza di misura, più un termine di prior:"); st.latex(r"J(\theta)=\sum_i\Big(\frac{S_i(\theta)-D_i}{u_D(D_i)}\Big)^2+\sum_j\Big(\frac{\theta_j-\theta_{j,0}}{s_j}\Big)^2")
    st.markdown("**Validazione.** Sui dati di test si calcolano RMSE, bias, MAE e $R^2$, e si confrontano con il rumore di misura:"); st.latex(r"\mathrm{RMSE}=\sqrt{\tfrac1n\sum_i(S_i-D_i)^2},\qquad R^2=1-\frac{\sum_i(S_i-D_i)^2}{\sum_i(D_i-\bar D)^2}")
    ui.callout("Le percentuali di casi «sotto soglia» descrivono l'incertezza di discretizzazione, non l'errore del modello rispetto alle misure: quest'ultimo si valuta con la calibrazione e la validazione.")
with t3:
    ui.passo(1, "<b>Cella, ricette e dati</b>: guarda la geometria, le conducibilità e le prove sperimentali; il modello del rumore alimenta i pesi della calibrazione.")
    ui.passo(2, "<b>Simulatore</b>: scegli ricetta, configurazione e tensione, avvia la simulazione e confronta con una prova. Il livello 1 (circa 2 300 nodi) richiede pochi secondi.")
    ui.passo(3, "<b>Verifica numerica</b>: esplora i risultati del protocollo (75 casi) e prova il raffinamento di mesh in diretta.")
    ui.passo(4, "<b>Sensitività</b>: esplora i risultati locali e globali per contrasto, tensione e configurazione, e calcola quelli locali per un caso a scelta.")
    ui.passo(5, "<b>Calibrazione</b>: anteprima in diretta con pochi gruppi e mesh grossolana, oppure importa i risultati del notebook di Colab.")
    ui.passo(6, "<b>Validazione</b>: analizza i risultati fuori campione, i residui e il confronto con il rumore di misura.")
    ui.passo(7, "<b>Modello nominale e misure</b>: prima di calibrare, confronta il modello con i parametri nominali con tutte le prove (bias per ricetta, tensione, configurazione e fase delle sonde).")
    ui.passo(8, "<b>Dossier del caso</b>: per una combinazione di ricetta, tensione e configurazione riunisce esiti di convergenza, passo temporale, sensitività, mesh suggerite e misure; la scheda si scarica.")
    ui.passo(9, "<b>Scenari e incertezza</b>: confronta condizioni operative, cerca la tensione minima che raggiunge una temperatura obiettivo e propaga l'incertezza dei parametri con un campione a ipercubo latino.")
    st.markdown("**Cosa gira nell'app e cosa no.** Le simulazioni singole, la verifica rapida, la sensitività locale in diretta e una calibrazione ridotta girano nell'app (secondi o minuti). I risultati del protocollo completo (75 casi con cinque livelli di mesh per due famiglie, la sensitività globale e la validazione incrociata su tutti i dati) richiedono ore e si eseguono con i notebook di Colab; l'app ne mostra i risultati già calcolati e permette di importare quelli di calibrazione e validazione.")
    st.markdown("**Mesh conformi.** Richiedono gmsh; se non è disponibile l'app usa solo le griglie strutturate.")
    st.markdown("**Limiti da tenere presenti.** Le bande di incertezza della pagina «Scenari e incertezza» comprendono solo l'incertezza dei parametri (intervalli da letteratura, non da misure), non quella numerica né quella di modello; la temperatura massima è la grandezza meno convergente del protocollo e va letta come indicazione del punto caldo; la calibrazione in diretta è un'anteprima con mesh grossolana e pochi gruppi.")
with t4:
    st.markdown("""
| Termine | Significato |
|:--|:--|
| Contrasto *c* | Rapporto σ_p/σ_m fra la conducibilità delle polpette e quella della purea a 20 °C |
| GCI | Indice di convergenza di griglia, stima dell'incertezza di discretizzazione con l'estrapolazione di Richardson |
| Regime monotono | I valori cambiano sempre nello stesso verso al raffinare la mesh |
| Π_p | Frazione della potenza elettrica dissipata nelle polpette |
| S_T, S_1 | Indice totale e del primo ordine di Sobol |
| μ* | Misura di Morris dell'effetto medio assoluto di un parametro |
| u_D | Incertezza di misura dell'incremento di temperatura |
| RMSE | Radice dell'errore quadratico medio fra simulato e misurato |
| Fold | Sottoinsieme delle condizioni usato per il test nella validazione incrociata |
""")
    st.markdown("**Parametri** (moltiplicatori del valore nominale)"); st.dataframe([{'Simbolo': dati.PAR_TXT[p], 'Descrizione': dati.PAR_DESC[p]} for p in dati.PAR], hide_index=True, width='stretch')
with t5:
    st.markdown("""
- Celik, I.B., Ghia, U., Roache, P.J., Freitas, C.J., Coleman, H., Raad, P.E. (2008). Procedure for estimation and reporting of uncertainty due to discretization in CFD applications. *Journal of Fluids Engineering* 130(7), 078001.
- Roache, P.J. (1994). Perspective: a method for uniform reporting of grid refinement studies. *Journal of Fluids Engineering* 116(3), 405–413.
- ASME (2009). *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*, ASME V&V 20-2009 (R2021).
- Eça, L., Dowding, K., Roache, P.J. (2020). On the interpretation and scope of the V&V 20 standard. *Verification & Validation Symposium*.
- Eurachem/CITAC (2012). *Quantifying Uncertainty in Analytical Measurement*, Guida CG 4, 3a ed.
- Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics* 33, 161–174.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions. *Reliability Engineering and System Safety* 93, 964–979.
- Palaniappan, S., Sastry, S.K. (1991). Electrical conductivity of selected juices: influences of temperature, solids content, applied voltage, and particle size. *Journal of Food Process Engineering* 14, 247–260.
- Sarang, S., Sastry, S.K., Knipe, L. (2008). Electrical conductivity of fruits and meats during ohmic heating. *Journal of Food Engineering* 87, 351–356.
""")
    st.caption("L'elenco completo, con le voci ancora da completare, è nella bibliografia della tesi.")
ui.piede()
