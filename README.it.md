# High-Precision Sports Ball Tracking

Rilevamento e tracciamento di un pallone da calcio in video, costruito per il caso difficile: un oggetto piccolo e veloce, spesso occluso, mosso o fuori campo. Il detector YOLO11s è volutamente semplice; la maggior parte dell'accuratezza viene da una pipeline di post-processing che trasforma le detection per-frame in una traiettoria fisicamente plausibile.

## Indice

- [Risultati](#risultati)
- [Come funziona](#come-funziona)
- [Decisioni](#decisioni)
- [Mappa del repository](#mappa-del-repository)
- [Eseguire i notebook](#eseguire-i-notebook)
- [Limiti](#limiti)
- [Documentazione](#documentazione)

## Risultati

L'obiettivo era un errore quadratico medio di 0,013 sulla posizione rispetto al ground truth.

| Metrica | Risultato |
| :--- | :---: |
| Box precision | 0,993 |
| Box recall | 0,968 |
| MSE finale | **0,0053** |
| MSE, clip 1 | 0,0078 |
| MSE, clip 2 | 0,0028 |

Il dato finale è la media sui due clip di test, ed è il numero su cui è stata selezionata la sequenza di post-processing.

## Come funziona

### Il detector

**YOLO11s**, scelto per il bilanciamento fra peso e calcolo, addestrato su Kaggle con una P-100.

Due modifiche mirano specificamente agli oggetti piccoli:

- la **testa P5 (layer 22) è congelata**, così la capacità resta sulle feature ad alta risoluzione che portano davvero un pallone largo 20 pixel;
- la risoluzione di training sale a **1088 px**, perché alla dimensione di default il pallone è una manciata di pixel dopo il primo downsampling.

L'augmentation (flip, rotazioni, variazioni di luminosità, rumore) è stata applicata tramite Roboflow.

### La pipeline di post-processing

La sola detection produce una nuvola di punti, non una traiettoria. La pipeline applica, in ordine:

1. **filtro outlier** — rimuove le detection incoerenti con il percorso recente, tramite distanza euclidea;
2. **interpolazione lineare** — riempie i buchi lasciati da occlusioni e frame mancati;
3. **traiettorie paraboliche** — ricostruisce i segmenti in cui il pallone esce dal campo visivo, il caso più debole del modello;
4. **validazione e smoothing** — controlla i segmenti interpolati prima che sostituiscano i dati reali.

L'ordine di questi passi è stato scelto empiricamente: è la sequenza che minimizza l'MSE sui clip di test, non quella che sembrava logica a priori.

### La valutazione

La metrica confronta la posizione predetta con il ground truth frame per frame, sui frame in cui esistono entrambi. La copertura sul primo clip fa parte del risultato:

| Misura | Conteggio | Quota |
| :--- | :---: | :---: |
| Indici di frame nell'intervallo | 2.564 | — |
| Frame con match (predizione e ground truth) | 237 | 68,5% dei frame con dati |
| Predizioni senza ground truth | 103 | 29,8% |
| Ground truth senza predizione | 6 | 1,7% |
| Indici di frame senza alcun dato | 2.219 | 86,5% dell'intervallo |

La maggior parte dell'intervallo non ha annotazioni, quindi il punteggio si calcola sul nucleo annotato del clip.

## Decisioni

**Un detector piccolo più post-processing pesante, invece di un modello più grande.** I modi di fallire del detector sono temporali — batti le palpebre e il pallone è perso — e nessun aumento di capacità ripara una traiettoria con un buco. Interpolazione e reiezione degli outlier recuperano quei frame a costo quasi nullo.

**Congelare la testa P5.** È una riduzione di capacità deliberata: per un oggetto da 20 pixel le feature più profonde sono rumore, e lasciarle addestrabili rallentava la convergenza senza migliorare il recall.

**Soglia di confidenza più bassa e tetto a 4 detection.** Normalmente un tracker tiene la detection più confidente. Qui il post-processing tiene quella **più vicina all'ultima valida**, perché una detection confidente può essere la testa di uno spettatore o una linea del campo. È una scelta che sposta il criterio di selezione dal detector alla traiettoria, dove sta la fisica.

**Ricostruzione parabolica per i palloni fuori campo.** Il modello non può vedere ciò che è uscito dall'inquadratura; un prior sulla traiettoria sì. È l'unica componente che aggiunge informazione che le immagini non contengono, ed è confinata ai segmenti in cui il pallone è fuori dal campo.

**Un MSE su misura invece della mAP come punteggio finale.** La mAP misura la qualità delle detection su box annotate; il prodotto è una posizione continua nel tempo. L'MSE sui frame con match è la metrica che riflette a cosa serve la pipeline.

## Mappa del repository

```
Notebook_Training_252053.ipynb   workflow di addestramento (Kaggle, GPU P-100)
Notebook_Test_252053.ipynb       inferenza, post-processing, valutazione MSE, video annotato
Relazione_Fuoco_Gabriele_252053.pdf   relazione tecnica: dataset, esperimenti, analisi degli errori
```

## Eseguire i notebook

```bash
pip install ultralytics roboflow opencv-python gdown pyyaml
```

I notebook sono scritti per Kaggle: all'inizio scaricano i pesi (`best.pt`, 19,3 MB) e i clip di test da Google Drive. **Pesi e video non sono in questo repository**, quindi un clone pulito ha bisogno che quei link siano raggiungibili prima di eseguire qualsiasi cosa.

## Limiti

| Cosa | Impatto | Stato |
|---|---|---|
| Pesi e video di test stanno su Google Drive | un clone pulito non esegue nulla senza di essi | aperto |
| Solo due clip di test | l'MSE è una media su due campioni, non un benchmark | accettato |
| Il 29,8% delle predizioni non ha ground truth | il punteggio non dice nulla su quei frame | aperto |
| Nessun test automatico | la pipeline si valida guardando grafici e video | aperto |
| Uno sport, un tipo di ripresa | movimento di camera, zoom e geometria del campo assumono il calcio | accettato |
| Box precision e recall vengono dalla validazione in training | non sono rimisurate dal notebook di test | annotato |

## Documentazione

`Relazione_Fuoco_Gabriele_252053.pdf` è la relazione tecnica: costruzione del dataset, i due esperimenti di training, il disegno del post-processing e l'analisi degli errori. I notebook sono la controparte eseguibile, con le celle di valutazione che hanno prodotto i numeri qui sopra.
