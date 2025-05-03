# Proposte classifier


## Preprocessing


### Fil

Concordo sul colore (irrilevante). Io più che scala di grigi penserei addirittura Binarizzazione fin da subito.
Ma oltre a quello che dice Chat non saprei proprio che altro dire (che Nicola non abbia già scritto.)

La Data Augmentation direi sia assolutamente necessaria; qualcosa come piccole rotazioni e flip dei diagrammi lungo gli
assi (tanto si scazza solo il testo che in teoria non è usato). PROTIP: Mettere anche nel dataset del classifier
diagrammi già compilati dei diversi tipi.


### Sav





### Nic

Il testo è (quasi, e.g. rettangolo del class diagram con testo) irrilevante nella classificazione.

Il colore è irrilevante. Anche se esistessero diagrammi in cui il colore ha un significato, 
comunque è improbabile che ci siano due tipologie di diagrammi con le stesse figure ma con colori diversi.

Le sole forme, il numero di forme e la dimensione delle forme non è sufficiente per distinguere due tipologie di diagrammi.
Si consideri per esempio class diagram e flow chart, entrambi hanno frecce e rettangoli.
Bisogna considerare le forme e le relazioni tra gli oggetti, senza la necessità di comprenderne il significato (che verrà fatto nell'estrattore).

Nelle immagini è presente rumore come ad esempio i quadretti del foglio, sbavature, immagini inclinate.


**Conversione in scala di grigi**

Ridurre la complessità computazionale eliminando le informazioni di colore superflue.

https://www.sciencedirect.com/science/article/pii/S2667305324000346


**Riduzione del Rumore e Filtraggio**

Rimuovere rumori, in teoria anche i quadretti del foglio con **filtro mediano** o **gaussiano**.

https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/ipr2.13243

**Binarizzazione e thresholding adattivo**

Semplificare la segmentazione e l’estrazione di caratteristiche

- Thresholding di Otsu
- Adaptive Mean / Gaussian Thresholding

https://www.sciencedirect.com/science/article/pii/S2667305324000346


**Allineamento**

Molti diagrammi sono scritti a mano in maniera inclinata.

Trasformata di Hough per rilevare l’angolo dominante e ruotare l’immagine di conseguenza.

Arrow R-CNN (Julca-Aguilar & Hirata, 2020)


**Ridimensionamento**

Ridimensionare tutte le immagini a 224x224 px (standard per CNN come ResNet)
oppure tenere una nostra dimensione.

Consigliano di normalizzare i valori dei pixel tra 0 e 1 con media 0 e varianza 1.



**Data augmentation**

Ridurre l’overfitting e migliorare la generalizzazione.

=> non genera i bbox

Handwritten Digit Recognition Using Deep Learning Algorithms, arXiv (2021)
https://arxiv.org/pdf/2106.12614




## Classifier

**Classifier** ("da zero"): Capire come fare il classifier (guardare paper di reti già fatte possibilmente su diagrammi)
  - Classi: `graph`, `flowchart`, `other` (circuiti, class diagram, diagrammi scolastici)
  - Dataset:
    - FA, FAB, ...
    - https://paperswithcode.com/dataset/ai2d
    - Circuiti
    - Class
    - BPMN

### Fil

Dopo aver guardato un po' in giro, pare proprio che le CNN funzionano bene per queste caratteristiche di task:

    1. Dataset piccolo (~1000 elementi)
    2. Riconoscimento di feature locali (e non di strutture globali)
    3. Molto più veloce di un ViT.

Il ViT potrebbe essere un'overkill - richiede almeno 100x la dimensione del dataset, ed è efficace per catturare 
relazioni globali all'interno dell'immagine.
Io proverei con una struttura di rete già rodata (tipo ResNet) provando a cambiare qualche valore nei layer o a
modificare cose semplici. Oppure partire da una CNN già rodata e capirne la filosofia, ma secondo me qualche paio
di layer convolutivo può essere sufficiente.


### Sav





### Nic

Dobbiamo "modificare" il dataloader del classificatore (o comunque lo script per l'assegnamento delle classi) affinché:

- Se l’etichetta originale è "flowchart" -> nuova etichetta "flowchart"
- Se l’etichetta originale è "graph" -> nuova etichetta "graph"
- Tutte le altre immagini (class diagram, circuiti, schemi scolastici, ...) -> nuova etichetta "other" 

Visto che abbiamo molte più immagini di "other" piuttosto che flowchart e graph dobbiamo fare data augmentation per oversampling:

- Flippare orizzontalmente le immagini
- Duplicare con un po' di rumore
- Sostituire il colore nero con altri colori (ci interessano solo le forme)


Reti pre-addestrate: ResNet18, ResNet50, EfficientNetB0 o MobileNetV2.

```python
import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
```

Possiamo cambiare solo l’ultimo strato per predire 3 classi.

Possiamo poi provare:

- Regressione lineare
- KNN
- Tree
- Fully connected

Fully connected forse è quella più **vendibile** come fatta da noi.