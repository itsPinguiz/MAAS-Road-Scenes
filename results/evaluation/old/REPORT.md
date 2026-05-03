# Fine-Grained Anomaly Detection Analysis: Results & Discussion

## Executive Summary
L'analisi dettagliata condotta sui modelli **ERFNet (CNN)** e **EoMT (Vision Transformer)** rivela vulnerabilità sistemiche che le metriche globali (come l'AuPRC totale) tendono a mascherare. Sebbene i modelli mostrino una discreta capacità di rilevamento per oggetti grandi e vicini, emergono tre criticità fondamentali per la sicurezza: 
1. **Crollo delle performance sui confini semantici** (boundary), dove l'incertezza aumenta drasticamente.
2. **Cecità quasi totale alle anomalie di piccola scala** e agli ostacoli all'orizzonte (B0-Far).
3. **Assorbimento tassonomico:** una forte tendenza a classificare le anomalie come "Road" o "Vegetation" (per EoMT) o classi in-distribution dominanti come "Truck" o "Train" (per ERFNet), neutralizzando l'OOD score.

---

## Task 1: The Boundary & Depth Problem (Space & Semantics)

Il rilevamento delle anomalie non è uniforme nello spazio dell'immagine. Abbiamo osservato una discrepanza sistematica tra le zone centrali degli oggetti ("Flat") e i loro bordi ("Boundary").

### Boundary Uncertainty
Per ERFNet nel dataset *RoadAnomaly21*, l'**FPR95** subisce un balzo critico: dal **61.94%** (Flat) al **90.43%** (Boundary) utilizzando il metodo MSP. Questo indica che quasi ogni pixel di confine viene erroneamente classificato o non rilevato con sufficiente confidenza, creando una "zona d'ombra" proprio dove la precisione del contorno è vitale per l'evitamento dell'ostacolo.

### Depth-Dependent Performance
La profondità di campo è il fattore più penalizzante. La "Vertical Blindness" è evidente osservando le fasce di profondità:
- **Fascia B0 (Far):** Entrambi i modelli sono virtualmente ciechi, con AuPRC che non superano il **9.58%** (EoMT).
- **Fascia B3 (Mid-Near):** Le performance raggiungono il picco (fino all'**85.38%** per EoMT), dimostrando che i modelli "reagiscono" solo quando l'ostacolo è ormai prossimo al veicolo.

| Model      | Region   | Method | AuPRC (%) | FPR95 (%) |
| :--------- | :------- | :----- | :-------: | :-------: |
| **ERFNet** | Boundary | MSP    |   56.38   |   90.43   |
| **ERFNet** | Flat     | MSP    |   25.06   |   61.94   |
| **EoMT**   | Boundary | MSP    |   92.59   |   44.91   |
| **EoMT**   | Flat     | MSP    |   63.22   |   30.26   |

![Boundary Delta ERFNet](analysis_reports/original/ERFNet/RoadAnomaly21/boundary_delta_road_anomaly21_ERFNet.png)
![Depth Heatmap EoMT](analysis_reports/original/EoMT/RoadAnomaly21/depth_heatmap_auprc_road_anomaly21_EoMT.png)

---

## Task 2: The Scale & Taxonomy Problem (Dimension & Type)

### The Scale Gap
Mentre i modelli CNN come ERFNet mantengono una minima (seppur insufficiente) sensibilità, i Vision Transformer (EoMT) mostrano un comportamento binary rispetto alla taglia:
- **Small Objects:** AuPRC prossimo allo **0.04%** o nullo (`nan`).
- **Large Objects:** AuPRC che sale al **67.7%**.
Questo suggerisce che il meccanismo di patching dei ViT, pur essendo potente per la comprensione globale, "filtra" via i dettagli ad alta frequenza necessari per identificare piccoli detriti stradali (es. mattoni, pneumatici).

### Misclassification & Absorption
L'analisi della **Class Prediction** su pixel OOD rivela come i modelli "anestetizzano" l'anomalia forzandola in classi familiari:
- **EoMT (RoadAnomaly21):** Predice il **57.6%** delle anomalie come **"Truck"**, indicando un bias verso oggetti massivi e rettangolari.
- **EoMT (Fishyscapes Static):** Assorbe il **49.4%** dei pixel OOD nella classe **"Road"** e il **27.3%** in **"Vegetation"**. L'anomalia scompare letteralmente nello sfondo semantico.
- **ERFNet:** Mostra una confusione più distribuita ma con picchi preoccupanti verso le classi **"Train" (26.2%)** e **"Car" (19.8%)**.

![AuPRC by Size EoMT](analysis_reports/original/EoMT/Fishyscapes_Static/size_auprc_fs_static_EoMT.png)
![Confusion Bar ERFNet](analysis_reports/original/ERFNet/RoadAnomaly21/confusion_bar_road_anomaly21_ERFNet.png)

---

## Task 3: Resolution Trade-off & Attention Mechanisms

### Resolution vs Performance
Esiste un evidente trade-off tra la risoluzione di input e la capacità di rilevamento. Aumentare la risoluzione migliora l'AuPRC sulle piccole scale, ma penalizza drasticamente i frame per secondo (FPS), rendendo il modello inutilizzabile in tempo reale.

### Attention Map Failure
L'ispezione delle mappe di attenzione di EoMT (basato su DINOv2) conferma il problema della scala. Nelle scene di *Fishyscapes*, i pesi dell'attenzione convergono sulle aree di "Vegetation" e "Building", ignorando completamente l'oggetto anomalo centrale se le sue dimensioni non sono sufficienti a influenzare i token globali. Questo spiega perché, nonostante la potenza del decoder, il segnale di incertezza (Entropy/MaxLogit) non viene innescato.

![FPS vs AuPRC](analysis_reports/original/ERFNet/RoadAnomaly21/auprc_vs_fps_road_anomaly21.png)
![Attention Map EoMT](analysis_reports/original/ERFNet/RoadAnomaly21/attention_overlay_fs_static.png)

---

## Conclusioni per la Ricerca

Questa analisi a grana fine dimostra che l'Anomaly Detection stradale non è un problema risolto. La dipendenza critica dalla **profondità** e dalla **scala** rende i modelli attuali inaffidabili per scenari di guida ad alta velocità, dove gli ostacoli devono essere rilevati quando sono ancora "piccoli" e "lontani". 

Le future direzioni di ricerca dovrebbero:
1. Sviluppare meccanismi di **Attention Multi-Scala** che preservino i token piccoli.
2. Implementare **Loss di Confine** specifiche per ridurre l'incertezza sui bordi semantici.
3. Superare il paradigma della "classificazione forzata", integrando decoder che segnalino esplicitamente la mancata corrispondenza con la distribuzione appresa (OOD-aware architectures).
