# Gemello digitale del riscaldamento ohmico multifase: app Streamlit

App interattiva sul caso di studio della cella di riscaldamento ohmico (99 × 54 × 66 mm, purea con polpette), costruita sopra il motore FEM 3D del protocollo di verifica, sensitività, calibrazione e validazione. Cinque ricette (0, 0.725, 1.45, 2.9 e 5.7% di sale), cinque configurazioni delle polpette (Tris, Symm, Asymm, Middle, Side), tre tensioni (30, 40 e 50 V), due famiglie di discretizzazione (griglie strutturate con interfaccia diffusa e mesh conformi con interfaccia netta).

## Pagine

| Gruppo | Pagina | Contenuto |
|:--|:--|:--|
| Panoramica | Panoramica | Caso di studio, indicatori principali e flusso degli studi |
| Cella e modello | Cella, ricette e dati | Geometria e configurazioni, conducibilità, esplorazione delle prove e modello del rumore |
| | Simulatore | Simulazione 3D accoppiata: sonde, campi, potenza ed energia, fasi, confronto con una prova |
| | Modello nominale e misure | Il modello con parametri nominali contro tutte le prove: bias e RMSE per ricetta, tensione, configurazione e fase |
| Studi numerici | Verifica numerica e incertezza | Convergenza di mesh e di passo temporale (GCI e minimi quadrati), esiti e soglie per grandezza, confronto fra famiglie, verifica rapida in diretta |
| | Sensitività | Locale e globale (Morris, Sobol), identificabilità, calcolo locale in diretta |
| | Dossier del caso | Tutti i risultati del protocollo per un singolo caso, con scheda scaricabile |
| Calibrazione e validazione | Calibrazione | Simulazioni dirette sulle prove, modelli annidati M1-M3, correlazione e incertezza dei parametri, diagnostica dei residui, importazione dei risultati di Colab |
| | Validazione | RMSE, bias, MAE, R² fuori campione, bootstrap per condizione, residui, stabilità dei parametri, validazione in diretta |
| Applicazioni | Scenari e incertezza | Confronto di scenari, tensione minima per una temperatura obiettivo, propagazione dell'incertezza dei parametri |
| Strumenti | Modello base monofase | Simulatore elettro-termico 3D di un alimento omogeneo |
| | Guida e metodi | Modello, criteri degli studi, uso dell'app, glossario e riferimenti |

## Avvio

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py          # oppure ./avvia.sh
```

La mesh conforme richiede `gmsh` (e, su Linux, le librerie elencate in `packages.txt`); se `gmsh` non è disponibile l'app usa soltanto le griglie strutturate. Su Streamlit Community Cloud basta caricare la cartella: `requirements.txt` e `packages.txt` sono letti automaticamente.

## Struttura

- `streamlit_app.py`: navigazione e tema.
- `pagine/`: una pagina per file.
- `core/`: dati (caricamento con cache), motore (simulazioni), analisi, calibrazione, scenari e componenti grafici.
- `engine/`: motore FEM e moduli del protocollo (gli stessi dei notebook di Colab).
- `data/`: risultati del protocollo e dati sperimentali, in formato compresso.
- `tests/`: collaudo con il framework di test di Streamlit.

## Collaudo

```bash
python tests/test_app.py                  # oppure: python -m pytest tests -q
```

Apre tutte le pagine ed esegue le funzioni in diretta con la mesh più grossolana (circa un minuto).

## Cosa gira nell'app e cosa no

Le simulazioni singole, la verifica rapida, la sensitività locale in diretta, il confronto con il modello nominale, gli scenari e una calibrazione ridotta girano nell'app (secondi o minuti). I risultati del protocollo completo (75 casi con cinque livelli di mesh per due famiglie, la sensitività globale e la validazione incrociata su tutti i dati) richiedono ore e si eseguono con i notebook di Colab: l'app ne mostra i risultati già calcolati e importa quelli di calibrazione e validazione (file `cv_*.json` e `previsioni_*.csv`).

## Limiti

- Le bande della pagina «Scenari e incertezza» comprendono solo l'incertezza dei parametri (intervalli da letteratura, non da misure), non quella numerica né quella di modello.
- La temperatura massima è la grandezza meno convergente del protocollo: è un'indicazione del punto caldo.
- Il modello non include l'ebollizione e si ferma a circa 90 °C.
- La calibrazione in diretta è un'anteprima, con mesh grossolana e pochi gruppi.
