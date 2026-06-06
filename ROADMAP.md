# Trading-Lab Roadmap Operativa

## Missione

Costruire una piattaforma quant modulare, terminal-first, orientata a ricerca, backtest, validazione, portfolio construction e operatività quotidiana. Il progetto nasce dalle 12 capability definite nelle slide: strategy generation, backtesting, risk-reward analysis, market regime detection, multi-factor strategy, optimization, portfolio construction, trade setup generation, Monte Carlo simulation, drawdown analysis, macro-based strategy e alpha/edge detection.[file:18]

Questa roadmap è il documento madre del progetto nello space. Serve per allineare architettura, priorità, naming, qualità esecutiva, uso della chat principale e criteri per aprire thread separati, così da evitare dispersione e confusione operativa.[file:18]

## Obiettivi pratici

Gli obiettivi del progetto sono tre:
- costruire un engine affidabile per ricerca e validazione di strategie;
- trasformare i risultati in workflow CLI ripetibili da terminale su macOS;
- mantenere ordine decisionale tra roadmap, issue GitHub, thread dedicati e implementazioni.[file:18]

Il principio guida è semplice: prima robustezza del core, poi velocità, poi sofisticazione. In pratica si implementano prima data layer, backtest e risk controls; solo dopo si estendono optimization, macro e alpha research.[file:18]

## Principi operativi

- Terminale come interfaccia primaria.
- Python come linguaggio principale.
- Repo GitHub come fonte di verità del codice.
- `ROADMAP.md` nello space come fonte di verità della direzione.
- Questa chat come cabina di regia del progetto.
- Thread secondari solo per lavori verticali o debugging intenso.[file:18]

## Problemi da correggere nel roadmap attuale

Il file attuale è già valido come base, ma ha alcuni punti da correggere per renderlo davvero esecutivo.[file:18]

- Manca una distinzione netta tra **core minimo funzionante** e moduli avanzati; questo rischia di diluire lo sforzo troppo presto.[file:18]
- Alcuni moduli sono descritti bene a livello concettuale ma senza definition of done, input/output e deliverable minimi.[file:18]
- C'è un refuso nei comandi CLI (`the`) che va eliminato per evitare copy-paste sporchi.[file:18]
- Non è ancora esplicitata una policy chiara su naming di file, cartelle, config, artifact e report.[file:18]
- Le regole per aprire nuovi thread sono buone, ma vanno trasformate in criteri operativi verificabili.[file:18]

## Struttura del repository

Repository consigliato: `trading-lab` sotto l'account GitHub `iFabbro`, con struttura iniziale minimale ma già pronta per crescere.[file:18]

```bash
mkdir -p trading-lab/{data,src,tests,scripts,config,notebooks,reports,docs}
cd trading-lab
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scipy numba ta yfinance matplotlib seaborn
pip install statsmodels scikit-learn pyyaml rich typer pytest
git init
```

Struttura consigliata:

```text
trading-lab/
├── ROADMAP.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── assets/
│   ├── strategies/
│   └── environments/
├── data/
│   ├── raw/
│   ├── cache/
│   ├── processed/
│   ├── backtests/
│   └── macro/
├── docs/
├── reports/
├── scripts/
├── src/
│   └── trading_lab/
└── tests/
```

## Convenzioni obbligatorie

### Naming

- Package Python: `trading_lab`
- File moduli: snake_case (`backtest_engine.py`, `regime_detection.py`)
- Script CLI: verbo + oggetto (`download_data.py`, `run_backtest.py`)
- Config strategy: `config/strategies/<strategy_name>.yaml`
- Report: `reports/<yyyy-mm-dd>_<topic>.md`
- Output backtest: `data/backtests/<strategy>__<asset>__<timeframe>.parquet`

### Git

- Branch principale: `main`
- Branch feature: `feat/<area>-<short-name>`
- Branch fix: `fix/<area>-<short-name>`
- Commit: imperativo breve, una responsabilità per commit
- Merge preferibilmente tramite PR, anche da solo, per lasciare traccia decisionale.[file:18]

### Qualità minima

Nessun modulo è “finito” se non soddisfa almeno questi criteri:
- eseguibile da CLI;
- test minimo presente;
- output salvato su file o stampato in modo leggibile;
- logica documentata in README o doc modulo;
- niente hardcode inutile di path, tickers o capitali.

## Macro-fasi del progetto

La roadmap originale copre 12 capability. Per eseguirla bene, il lavoro va riorganizzato in 5 macro-fasi.[file:18]

| Macro-fase | Scopo | Moduli coinvolti |
|---|---|---|
| Fase A | Core ricerca e dati | 1, 2 |
| Fase B | Validazione rischio | 3, 9, 10 |
| Fase C | Adattamento al mercato | 4, 6 |
| Fase D | Produzione operativa | 7, 8 |
| Fase E | Ricerca avanzata | 5, 11, 12 |

## Fase A — Core ricerca e dati

### A1. Data layer

Obiettivo: avere un layer dati unico, ripetibile, testabile e cache-aware.[file:18]

**Deliverable minimi**
- `load_ohlcv()` funzionante
- cache locale in `data/raw/` e `data/cache/`
- supporto almeno a 1 asset crypto e 1 equity
- script CLI `download_data.py`
- 2 test base su shape, colonne e date

**Definition of done**
- scarichi dati da CLI senza modificare codice;
- riesegui lo stesso comando e il caching evita lavoro inutile;
- il dataframe ha schema standardizzato (`open`, `high`, `low`, `close`, `volume`, `timestamp`).

### A2. Strategy generation

Obiettivo: definire strategie parametriche in modo standard e componibile.[file:18]

**Deliverable minimi**
- dataclass `StrategyConfig`
- 3 strategie base codificate
- supporto a parametri indicatori, entry/exit, SL/TP, sizing
- salvataggio config in YAML

**Definition of done**
- una strategia può essere definita senza toccare il motore di backtest;
- puoi caricare una strategia da file config;
- ogni strategia produce segnali riproducibili.

### A3. Backtest engine

Obiettivo: eseguire backtest seri e ripetibili su 5–10 anni, con metriche fondamentali.[file:18]

**Deliverable minimi**
- `BacktestResult`
- equity curve
- trade log
- metriche: CAGR, Sharpe, max drawdown, win rate
- script `run_backtest.py`

**Definition of done**
- il comando CLI gira da zero;
- i risultati vengono salvati in `data/backtests/`;
- lo stesso input produce lo stesso output, a parità di dati e config.

## Fase B — Validazione rischio

### B1. Risk-reward analysis

Obiettivo: misurare qualità economica dei trade e struttura del rischio.[file:18]

**Deliverable minimi**
- distribuzione dei trade in R
- average R, payoff ratio, losing streak
- report markdown salvato in `reports/`

### B2. Monte Carlo

Obiettivo: stressare la sequenza dei trade e stimare fragilità/robustezza.[file:18]

**Deliverable minimi**
- simulazioni su trade reshuffling o bootstrap
- probabilità di chiudere in perdita
- quantili worst-case su ritorno e drawdown

### B3. Drawdown analysis

Obiettivo: capire quanto fa male il sistema quando smette di funzionare bene.[file:18]

**Deliverable minimi**
- max drawdown
- drawdown duration
- average recovery time
- 3 leve pratiche per ridurre DD

**Gate di avanzamento**

Non si passa alla fase successiva se una strategia non ha:
- backtest consistente;
- report rischio leggibile;
- Monte Carlo di base;
- analisi drawdown completata.

## Fase C — Adattamento al mercato

### C1. Market regime detection

Obiettivo: etichettare il contesto di mercato e capire quando una strategia ha o perde edge.[file:18]

**Deliverable minimi**
- classificazione bull/bear/sideways
- classificazione volatility high/normal/low
- classificazione volume strong/weak
- funzione di raccomandazione strategica

### C2. Strategy optimization

Obiettivo: migliorare strategie esistenti senza overfittare.[file:18]

**Deliverable minimi**
- `optimization.py`
- grid search o random search
- confronto baseline vs optimized
- filtri contro overfitting: trade minimi, stabilità cross-period, limiti di complessità

**Regola importante**

Ogni ottimizzazione va validata su almeno due segmenti: in-sample e out-of-sample. Se migliora solo in-sample, non è promossa a strategia operativa.

## Fase D — Produzione operativa

### D1. Portfolio construction

Obiettivo: passare da singole strategie a una struttura allocativa gestibile.[file:18]

**Deliverable minimi**
- equal weight
- risk parity semplice
- mean-variance con vincoli minimi
- report pesi, rischio e motivazione per asset

### D2. Trade setup generation

Obiettivo: generare setup giornalieri o intraday leggibili e usabili.[file:18]

**Deliverable minimi**
- 3 trade setup prioritizzati
- entry, stop, target, RR
- ragione tecnica e di contesto
- stampa CLI leggibile con `rich`

**Gate di produzione**

Un setup non esce in produzione se non deriva da:
- strategia versionata;
- dati aggiornati;
- regole di rischio definite;
- regime compatibile.

## Fase E — Ricerca avanzata

### E1. Multi-factor strategy

Questa capability ha senso solo quando il core single-strategy è stabile. Prima si prova la macchina, poi si aggiunge complessità cross-sectional.[file:18]

### E2. Macro-based strategy

La componente macro va trattata come layer di allocazione o filtro, non come scorciatoia narrativa. Le serie macro possono essere raccolte con Google/Antigravity e salvate localmente in `data/macro/` per risparmiare crediti e mantenere controllo sulla pipeline.[file:18]

### E3. Alpha / edge detection

Questo è il modulo più sperimentale e va isolato in thread dedicati. Richiede ricerca, raccolta evidenze, prototipi veloci e test separati dal core, per evitare che il progetto principale si trasformi in un laboratorio caotico.[file:18]

## Ordine reale di implementazione

L'ordine corretto non è 1→12 in linea retta, ma questo:

1. Data layer
2. Strategy config
3. Backtest engine
4. Risk report
5. Drawdown analysis
6. Monte Carlo
7. Regime detection
8. Optimization
9. Portfolio construction
10. Trade setup generation
11. Multi-factor
12. Macro
13. Alpha detection

Questo ordine è più sicuro perché costruisce prima il motore di verifica, poi l'adattamento, poi la parte operativa e infine la ricerca avanzata.[file:18]

## Workflow da terminale

L'uso corretto del progetto deve essere completamente CLI-first.

```bash
python scripts/download_data.py --ticker BTC-USD --start 2018-01-01 --timeframe 1d
python scripts/run_backtest.py --strategy-id ma_crossover --ticker BTC-USD --start 2018-01-01 --end 2026-01-01 --timeframe 1d --capital 10000
python scripts/analyze_risk.py --strategy-id ma_crossover --ticker BTC-USD
python scripts/run_montecarlo.py --strategy-id ma_crossover --ticker BTC-USD
python scripts/generate_trades.py --market crypto --ticker BTC-USD
```

Questa sezione sostituisce la versione con il refuso `the`, che va rimosso dal file originale.[file:18]

## GitHub workflow

Il tuo GitHub è `iFabbro`, quindi la disciplina del repo va impostata bene fin dall'inizio: roadmap nel repo, issue per fasi, branch corti e PR con scopo singolo. Questo riduce la confusione e crea una memoria tecnica persistente fuori dalla chat.[file:18]

**Schema minimo consigliato**
- Issue 1: bootstrap repo e struttura base
- Issue 2: data layer
- Issue 3: backtest engine
- Issue 4: risk + drawdown
- Issue 5: Monte Carlo
- Issue 6: regime detection
- Issue 7: optimization
- Issue 8: portfolio + trade setup
- Issue 9: macro layer
- Issue 10: alpha research

## Uso della chat principale

Questa chat è il centro di coordinamento. Qui devono restare solo contenuti ad alta leva decisionale.[file:18]

**In questa chat si fa:**
- roadmap e priorità;
- decisioni architetturali;
- naming e convenzioni;
- scelta del prossimo blocco da implementare;
- revisione dello stato generale del progetto.

**In questa chat non si dovrebbe fare per troppo tempo:**
- debugging lungo di un singolo script;
- analisi dettagliata di stacktrace multipli;
- esperimenti macro/alpha scollegati dal core;
- editing massivo di un modulo isolato.

## Quando aprire un nuovo thread

Apri un nuovo thread quando si verifica almeno una di queste condizioni:
- una singola feature richiede più di 20–30 messaggi tecnici consecutivi;
- stai lavorando su un solo modulo senza impatto immediato sugli altri;
- hai bisogno di incollare molto codice, log o test output;
- stai facendo debug intensivo;
- stai sperimentando idee non ancora promosse nella roadmap principale.[file:18]

Aprire un nuovo thread **non è dispersione**; è una misura di igiene cognitiva del progetto. Il thread principale deve restare leggibile anche tra settimane.[file:18]

## Come aprire correttamente un thread secondario

All'inizio del nuovo thread incolla sempre:
- nome repo;
- link GitHub;
- riferimento al `ROADMAP.md` nello space/repo;
- fase o modulo coinvolto;
- obiettivo preciso del thread;
- stato attuale e blocco tecnico.

Template consigliato:

```text
Repo: trading-lab
GitHub: https://github.com/iFabbro
Documento guida: ROADMAP.md
Fase: Backtest Engine
Obiettivo thread: implementare trade loop + metriche base
Contesto: il data layer è già pronto, ora serve il primo motore funzionante
Blocco attuale: definizione schema Trade e BacktestResult
```

## Regola di rientro nel thread principale

Quando chiudi un thread secondario, torna qui e registra solo 4 cose:
- cosa è stato completato;
- cosa è cambiato rispetto alla roadmap;
- cosa resta aperto;
- quale sarà il prossimo thread, se serve.

In questo modo la chat principale resta un changelog strategico, non un contenitore disordinato di tutto.[file:18]

## Uso di Antigravity e Google

Per risparmiare crediti, Antigravity/Google vanno usati per ricerca veloce su:
- documentazione di librerie;
- formule tecniche standard;
- definizioni macro;
- esempi di implementazione non proprietari;
- raccolta manuale di serie macro o metadata.[file:18]

La chat va usata invece per:
- trasformare ricerca grezza in architettura concreta;
- integrare componenti nel progetto;
- progettare interfacce, modelli dati e workflow;
- revisionare decisioni e priorità.[file:18]

Regola pratica: se una domanda si risolve bene in 2 minuti di browser, va fatta fuori dalla chat; se richiede integrazione, trade-off o struttura, va fatta qui.

## Checklist di maturità del progetto

### Livello 1 — Bootstrap
- repo creato
- roadmap presente
- ambiente Python pronto
- struttura directory pronta

### Livello 2 — Core affidabile
- data layer stabile
- strategy config stabile
- backtest funzionante
- metriche salvate

### Livello 3 — Risk-aware
- risk report attivo
- drawdown report attivo
- Monte Carlo attivo

### Livello 4 — Operativo
- regime detection attiva
- portfolio logic attiva
- setup generator attivo

### Livello 5 — Ricerca avanzata
- optimizer robusto
- modulo macro usabile
- alpha research isolata e documentata

## Prime 3 azioni adesso

1. Pulire il file `ROADMAP.md` originale sostituendolo con questa versione revisionata.
2. Creare il repo se non esiste ancora e aprire le prime 3 issue: bootstrap, data layer, backtest engine.
3. Aprire un thread secondario dedicato solo alla Fase A1-A3 quando inizi a scrivere codice vero.

## Decisione operativa

Da questo momento, questa chat resta il thread madre del progetto. Ogni volta che il lavoro diventa verticale, rumoroso o troppo locale a un modulo, si apre un nuovo thread; quando il blocco è risolto, si torna qui per aggiornare la direzione generale.[file:18]
