
# ⚽ Soccer Ball Tracking: High-Precision Sports Ball Tracking

Questo progetto implementa un sistema avanzato di **object detection** e **tracciamento** specializzato nel rilevamento di palloni da calcio in sequenze video, ottimizzato per piccoli oggetti e situazioni di movimento rapido.

## 🚀 Panoramica del Progetto

Il sistema è stato sviluppato utilizzando **YOLO11s** come modello base, addestrato in ambiente Kaggle e raffinato tramite sofisticate tecniche di **post-processing** per garantire traiettorie fluide e precise.

## 🛠️ Caratteristiche Principali

- **Modello YOLO11s**: Scelto per l'ottimo bilanciamento tra leggerezza ed efficienza computazionale.
    
- **Ottimizzazione "Small Object"**: Congelamento della testa P5 (livello 22) per concentrare le risorse sugli oggetti piccoli e aumento della risoluzione di training a **1088px**.
    
- **Data Augmentation**: Utilizzo di Roboflow per applicare flip, rotazioni, variazioni di luminosità e rumore.
    
- **Pipeline di Post-Processing**:
    
    - ✨ **Filtro Outlier**: Rimozione di rilevamenti errati basata su distanza euclidea.
        
    - 📈 **Interpolazione Lineare**: Riempimento dei gap temporali (occlusioni).
        
    - 🏹 **Traiettorie Paraboliche**: Ricostruzione predittiva delle traiettorie di uscita dal campo.
        

## 📊 Performance

Il sistema ha superato l'obiettivo di MSE (Mean Squared Error) fissato a 0.013:

|**Metrica**|**Risultato Finale**|
|---|---|
|**Box Precision**|0.993|
|**Box Recall**|0.968|
|**Final MSE**|**0.0053**|

## 📂 Struttura del Repository

- `Notebook_Training.ipynb`: Workflow completo di addestramento su GPU P-100.
    
- `Notebook_Test.ipynb`: Script per l'inferenza, valutazione MSE e generazione video tracciati.
    
- `Relazione_Fuoco_Gabriele.pdf`: Documentazione tecnica dettagliata.
    

## ⚙️ Requisiti

Bash

```
pip install ultralytics roboflow opencv-python gdown pyyaml
```

---

**Autore:** Gabriele Fuoco