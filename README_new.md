# D.I.A.G.R.A.M.: Development of Image Analysis for Graph Recognition And Modeling

This project proposes the development of a system for analyzing different types of handwritten diagrams and converting them into well-rendered images through a textual syntax.
The goal is to create a tool capable of analyzing scanned or photographed sketches of diagrams and automatically generating code that can be rendered into the same diagrams digitally, 
to eventually integrate or modify them.


## Get Started

### Install

1. Installa **D2 CLI** dal repository ufficiale: [https://github.com/terrastruct/d2](https://github.com/terrastruct/d2) (cercare la versione appropriata per il proprio sistema nelle [releases](https://github.com/terrastruct/d2/releases))

Dopo aver installato la CLI dovresti essere in grado di eseguire `d2`.

```bash
d2 --version
```

2. Installare le librerie necessarie (eventualmente in un ambiente Conda)

```bash
pip3 install torch opencv-python matplotlib requests pillow pandas torchvision numpy shapely transformers sentencepiece protobuf torchmetrics scikit-learn
```




## Project Overview

Il sistema si compone di diverse parti:

- **Preprocessor**: preprocessa le immagini, e.g. raddrizza le immagini (*geometria*)
- **Classifier**: classifica i diagrammi restituendo la tipologia (e.g. `graph-diagram`, `flow-chart`)
- **Extractor**: entità astratta per *estrarre* una rappresentazione agnostica di *una specifica tipologia* di diagramma da un'immagine (e.g. matrice del grafo per i `graph-diagram`)
- **Transducer**: traduce (staticamente) i concetti agnostici in uno *specifico* linguaggio di markup di rappresentazione dei diagrammi (e.g. Mermaid)
- **Compiler**: compila il linguaggio di markup nell'effettivo diagramma
- **Orchestrator**: gestisce il flusso e i componenti

![Overview](doc/assets/images/overview.png)

La rete classificatrice è utilizzata per individuare quale modulo estrattivo utilizzare.

Ogni extractor è **specializzato** su una sola tipologia di digramma.

Per esempio, dato come input un'immagine di un grafo:

![Input](dataset/source/fa/test/writer018_fa_001.png)

Il classificatore produce `graph-diagram`, dunque l'orchestratore porta all'extractor per i diagrammi l'immagine di input.

L'extractor dei diagrammi produce la matrice del grafo, dove per righe e per colonne si hanno i **nodi** e i **fuori nodi** (per gestire frecce che partono dal nulla). Il valore è un interno non negativo che indica il numero di connessioni (la posizione nella matrice indica provenienza e destinazione). Inoltre, produce le datastruct di lookup per notazioni sulle frecce e testo dei nodi.

L'orchestratore porta in input dei trasduttori (in base input dell'utente oppure tutti) della relativa tipologia di diagramma, i quali produrranno le traduzioni in linguaggio di lookup.
Per esempio, in Mermaid si potrebbe avere del testo del tipo:

```
graph LR;
    q0--> q0 & q1
    q1--> q2
    q2--> q2 & q1
```

Infine, il linguaggio di markup è compilato con il relativo compilatore.



## Dataset

Source: https://github.com/bernhardschaefer/handwritten-diagram-datasets


### Graph diagram: dataset/source/fa

We used the following datasets:
- https://cmp.felk.cvut.cz/~breslmar/finite_automata/ (**graph diagrams** only, fa)
- https://cmp.felk.cvut.cz/~breslmar/flowcharts/ (**flowchart diagrams**, fcb)
- https://tc11.cvc.uab.es/datasets/OHFCD_1 (**flowchart diagrams**, fca)
- https://github.com/dwslab/hdBPMN (**BPMN diagrams**, hdBPMN)
- https://www.kaggle.com/datasets/leticiapiucco/handwritten-uml-class-diagrams (**class diagrams**)
- https://github.com/aaanthonyyy/CircuitNet (**circuit diagrams**)
- https://paperswithcode.com/dataset/ai2d (**school diagrams**)

For the classifier, we used the class, circuit and school diagrams to gain more robustness

For the extractor, we used only flowchart and graph diagrams

The diagrams are all **annotated** with **bounding box**, *also for the text*
- in addition to the provided bounding boxes for the arrows we added the bounding boxes for the head(s) and tail(s)
  - we assumed an arrow can have two heads, two tails, or one head and one tail

*Categories*

![Categorie](doc/assets/images/categories-fa.png)

*Annotation example*

![Esempio annotazione](doc/assets/images/annotation-fa.png)
