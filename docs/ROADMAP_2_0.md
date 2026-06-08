# Trading-Lab Roadmap 2.0

## Obiettivo generale
Portare Trading-Lab da piattaforma modulare funzionante a sistema operativo personale affidabile e, in prospettiva, a motore di trading autonomo capace di monitorare il mercato, generare segnali, validare il rischio ed eseguire ordini in modo controllato.

## Principio guida
La priorità non è solo aggiungere moduli, ma:
1. rendere il sistema robusto;
2. migliorare la qualità dei dati e degli output;
3. semplificare l’interfaccia operativa;
4. separare chiaramente monitoraggio, decisione ed execution;
5. preparare il progetto a una futura espansione controllata.

## Architettura logica target
Il sistema finale deve essere composto da 5 layer principali:
- **Research layer**: generazione strategie, backtest, ottimizzazione.
- **Decision layer**: segnali, regime detection, portfolio logic.
- **Risk layer**: sizing, drawdown, Monte Carlo, limiti di esposizione.
- **Execution layer**: apertura, modifica e chiusura ordini.
- **Monitor layer**: HUD live, report, alert, stato operativo.

## Fase A — Stabilizzazione base
### Obiettivo
Rendere affidabili i moduli già attivi.

### Attività
- Standardizzare i formati CSV/JSON prodotti dai vari script.
- Definire uno schema unico per trade log, metrics, equity curve e setup operativi.
- Aggiungere controlli minimi di validazione dati.
- Rendere il backtest più difendibile dal punto di vista numerico e documentare i limiti della strategia demo.

### Output atteso
- Output coerenti tra script.
- File leggibili e uniformi.
- Minore dipendenza da input “fortuiti”.

## Fase B — Dashboard 2.0
### Obiettivo
Trasformare la HUD terminale in un vero cockpit operativo personale.

### Attività
- Ridurre la densità visiva.
- Separare chiaramente:
  - stato sistema,
  - KPI operativi,
  - trade aperte,
  - alert,
  - snapshot macro/report.
- Introdurre priorità visive.
- Rendere il refresh stabile in-place.
- Aggiungere modalità:
  - snapshot;
  - live;
  - minimal.

### Output atteso
- HUD compatta.
- Lettura in 2 secondi.
- Informazioni critiche sempre nello stesso punto.

## Fase C — Live data layer
### Obiettivo
Passare da snapshot locali a flusso più affidabile.

### Attività
- Definire un layer dati più robusto.
- Introdurre refresh controllato dei file operativi.
- Aggiungere verifiche di freschezza dati.
- Separare chiaramente dati storici, cache e stato operativo.

### Output atteso
- Dati più stabili.
- Monitor più affidabile.
- Riduzione dei mismatch tra file e dashboard.

## Fase D — Trade engine usability
### Obiettivo
Rendere il sistema più utile per l’operatività quotidiana.

### Attività
- Standardizzare trade aperte/chiuse.
- Aggiungere metriche operative:
  - PnL realizzato,
  - PnL non realizzato,
  - esposizione long/short,
  - distanza dallo stop.
- Migliorare il parsing dei setup.
- Rendere i report più coerenti.

### Output atteso
- Lettura immediata dello stato posizioni.
- Alert più chiari.
- Uso più pratico durante la giornata.

## Fase E — Backtest quality
### Obiettivo
Migliorare la qualità quantitativa del motore di test.

### Attività
- Introdurre strategie demo più realistiche.
- Separare chiaramente motore e strategia.
- Validare meglio posizione, sizing e PnL.
- Aggiungere test di regressione sulle metriche principali.

### Output atteso
- Backtest più credibile.
- Maggiore utilità per valutazione comparativa.
- Meno rischio di interpretare male i risultati.

## Fase F — Risk layer avanzato
### Obiettivo
Rendere il sistema utile anche come strumento di controllo del rischio.

### Attività
- Rafforzare risk analysis e drawdown.
- Collegare i report al monitor live.
- Aggiungere soglie di warning operative.
- Integrare scenari Monte Carlo più leggibili.

### Output atteso
- Maggiore capacità di controllo.
- Migliore lettura della fragilità strategica.
- Alert rischio più utili.

## Fase G — Execution autonoma
### Obiettivo
Introdurre un layer separato per generare ed eseguire ordini in autonomia, senza sovrapporlo alla HUD.

### Attività
- Creare `src/execution.py` per la logica di execution.
- Creare `scripts/run_bot.py` come entrypoint operativo del motore autonomo.
- Definire input chiari per:
  - segnali;
  - risk check;
  - position sizing;
  - ordine di mercato / limit / stop;
  - conferma esecuzione.
- Separare la decisione dall’invio ordini.
- Introdurre logica di retry, fallback e gestione errori.
- Collegare il layer di execution al rischio e allo stato posizione.
- Preparare interfacce future per broker/API.

### Output atteso
- Motore capace di aprire, modificare e chiudere ordini secondo regole predefinite.
- Log strutturato delle operazioni.
- HUD che resta solo monitor, non execution.

## Fase H — Portfolio and regime
### Obiettivo
Collegare il motore operativo alla logica di allocazione e al contesto di mercato.

### Attività
- Raffinare portfolio construction.
- Collegare il regime macro alle esposizioni.
- Introdurre logiche più esplicite di allocazione e disallocazione.
- Rafforzare il legame tra macro regime e setup suggeriti.
- Integrare la regime awareness nel layer di decisione.

### Output atteso
- Sistema più coerente con il contesto di mercato.
- Minore uso “piatto” delle strategie.
- Maggiore adattività.

## Fase I — Optimization layer
### Obiettivo
Preparare il progetto a ottimizzazioni controllate.

### Attività
- Creare un modulo di ottimizzazione più formalizzato.
- Definire griglie e criteri di selezione.
- Evitare overfitting eccessivo.
- Documentare le metriche di scelta.

### Output atteso
- Parametri più difendibili.
- Migliore tracciabilità delle scelte.
- Processo replicabile.

## Fase J — Execution safety and supervision
### Obiettivo
Assicurare che il motore autonomo operi con vincoli chiari, supervisionabili e interrompibili.

### Attività
- Introduzione di kill switch.
- Limiti giornalieri di perdita.
- Limiti di numero trade.
- Max exposure per simbolo e per portafoglio.
- Modalità paper / live / dry-run.
- Audit log completo delle decisioni.
- Stato persistente del bot.

### Output atteso
- Motore autonomo più sicuro.
- Controllo umano sempre possibile.
- Migliore governance dell’automazione.

## Fase K — Future extensions
### Obiettivo
Aggiungere funzionalità solo dopo consolidamento.

### Possibili estensioni
- watchlist multi-asset più evoluta;
- alert intelligenti;
- logging più strutturato;
- export report automatici giornalieri;
- integrazione con notifiche esterne;
- analisi multi-timeframe;
- miglioramenti visuali della HUD;
- strategia macro più sofisticata;
- alpha detection più profonda;
- execution multi-broker;
- portfolio automation più avanzata.

## Priorità di esecuzione
1. Dashboard 2.0.
2. Stabilizzazione del data/output layer.
3. Standardizzazione trade e report.
4. Miglioramento backtest e risk.
5. Execution autonoma.
6. Rafforzamento del layer portfolio/macro.
7. Safety, supervision e future extensions.