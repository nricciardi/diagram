# TODO

## Lunedì 31/03

- [x] ~~Rappresentazione universale~~
- [ ] Definire una nuova tipologia di diagramma
- [x] Trasduttori Mermaid e PlantUML per graph diagram e flow chart
- [x] Orchestratore sequenziale

## Lunedì 07/04

- [ ] Scegliere un nuovo diagramma
- [x] ~~fil: Refactor (non necessario) transducer (usare le librerie sul gruppo)~~
- [x] sav: Wrappare compilatore Mermaid (https://mermaid.js.org/ecosystem/tutorials.html#jupyter-python-integration-with-mermaid-js)
- [x] sav: Wrappare compilatore D2
- [x] nic: Multiprocess orchestratore
- [x] fil: Proposal


## Lunedì 14/04

- [x] nic: CLI per main
- [x] fil: Rollback transducer
- [x] sav: Definire l'output/input della rete dei bbox dell'extractor (07/04/2025 alle 10)
- [x] fil: Refactor DiagramRepresentation
- [x] fil: Refactor TransducerMermaid
- [x] fil: Refactor TransducerD2
- [x] fil: Test completo transducer + compiler
- [x] nic: Template pattern extractor
- [x] nic: Definire output tra punto 3 e 4 extractor
- [x] sav: Portare proposte dataloader
- [x] nic: mail a Baraldi

## Martedì 22/04

- [x] nic: Filtro confidenza
- [x] sav: Dataset (`DataLoader`)
  - [x] Parse del dataset per i nostri scopi
  - [x] Fillare la classe wrapper `Image` per avere sia il "content opencv" sia annotazioni "json" come attributi (e.g. `TensorImage` o qualcosa di simile)
  - [x] Fillare TODO main.py
  - [x] Creare classe per gestire il dataset (e.g. get_image); guardare se c'è la classe
- [x] nic: Associare i testi alle frecce e ai nodi
- [x] fil: Overlapping nodi-frecce per relazioni
- [x] sav: Overlapping nodi-testo
- [x] sav: Overlapping frecce-testo
- [x] fil: Creazione DiagramRepresentation
- [x] nic: Gestione del secchio degli scartati
- [x] fil: Digitalizzazione del testo (rete già fatta)
- [x] fil: Testare la digitalizzazione del testo
- [x] sav: print -> log
- [x] test Image, TensorImage
- [x] sav: Aggiungere doc utils
- [x] sav: Accorpare utils bbox in un solo file `bbox` (?)

**Dottorando deve rispondere**

- [ ] Organizzare deadline

## Lunedì 28/04


   G   N   R
- [ ] [x] [ ] **Ottenere rappresentazione freccia** (as-is) (retta passante per test* o cod* da cui ottenere bbox 4 punti): provare [rete](https://link.springer.com/article/10.1007/s10032-020-00361-1) oppure trovarne un'altra (oppure piangere e farla noi)
  - [ ] Cercare testa e coda delle frecce (rete) [sgherro]
- [x] [x] [x] Aggiornare docstring
- [x] fil: refactor distanza in creazione relazioni
- [x] sav: togliere direzioni
- [x] fil: togliere direzioni
- [x] ipotesi/assunzioni, a linguaggio naturale come funziona, problemi e idee (8 direzione/posizione/... freccia)
  - [ ] Aggiornare il readme (assunzioni fatte, struttura overview sistema, estrattore con tutti i passaggi), rimuovere i file md inutili
- [x] [ ] [x] **Classifier** ("da zero"): Capire come fare il classifier (guardare paper di reti già fatte possibilmente su diagrammi)
  - Classi: `graph`, `flowchart`, `other` (circuiti, class diagram, diagrammi scolastici)
  - Dataset:
    - FA, FAB, ...
    - https://paperswithcode.com/dataset/ai2d
    - Circuiti
    - Class
    - BPMN
- [ ] [ ] [ ] **Object detection** (fine-tuning): trovare reti già fatte (possibilmente sui diagrammi o simili)
  - Dataset: FA, FAB, ...
- [x] nic: mandare mail allo sgherro
- [x] sav: complete classifier dataset


Per lo sgherro (in base alla priorità):

1. Problema delle frecce
2. Object detection
3. Classifier
4. Preprocessing
5. (Dataset)


## Lunedì 12/05

- [ ] nic: Doppio clustering
- [ ] nic: riportare cose dette con zini
- sav[x] fil[x] togliere pianto.md previo controllo
- [x] fil: paper (abstract, introduzione, architettura(classifier(preprocessing, ...), extractor(preprocessing, bbox, ...), transducer, compiler))
- [ ] object detection extractor
  - [x] sav (linka la guida): Capire come far funzionare i nodi con vscode
    - [Guide](https://ailb-web.ing.unimore.it/coldfront/documentation/Zf44P) + Gibbo's file
  - [x] sav: Far funzionare dataloader per object detection
  - [ ] sav: Primo esperimento import torchvision e allenare _da zero_ sui nostri dati
  - [ ] Secondo esperimento import torchvision e non allenare da zero (qualche resnet) -> fine tuning












## Lunedì 05/05

- [x] Preprocessing extractor ~~[sgherro]~~ GRANDE FILLO
- [ ] Object detection -> nodi, frecce, testo [sgherro]
- [ ] Classifier
  - [ ] Preprocessing ~~[sgherro]~~ GRANDE FILLO
    - [x] Definire quali fare -> chiedendo al dottorando
    - [x] Togliere rumore (gaussian filter) [filter]
  - [x] Classifier: Raddrizzare immagini [geom]
  - [x] Togliere il background (e.g. togliere quadretti) [e.g. Otsu]
  - [ ] Rete classificatrice
    - [x] Definire -> ~~chiedendo al dottorando~~


## Domenica 22/06

- [x] F: Master guide
- [x] F. Downsampling/upsampling classifier
- [x] F: Classi prese in input dal classifier
- [x] F: print -> log
- [x] N: Identificazione testa e coda frecce da bbox 2 punti
- [x] S: Finetuning object detection singola senza Otsu e filtro mediano (se va male farne due)


Scegliere se andare a fare spionaggio


## Domenica 29/06

- [ ] Fil: _compute_relations in extractor
- [ ] Sav: _arrow_text_type in extractor
- [ ] Chat: Finetuning threshold extractor
- [x] N: Fix extractor (tra cui aggiungere i metodi astratti per transduer e extractor)
- [ ] Togliere i TODO nel progetto, togliere percorsi assoluti
- [ ] F: Spostare preprocessor
- [ ] N: Non va debug nei log
- [ ] F: unsqueeze sul preprocessor grayscale
- [ ] F: `"cuda" if torch.cuda.is_available() else "cpu"` nei text extractor -> usare to_device mixin (nostro)


## Deadline codice 01/07

- [ ] Testing

## Deadline paper e presentazione 07/07

- [ ] Paper
- [ ] Presentazione

## Deadline e presa della Bastiglia 14/07

- [ ] Presentazione orale















## Backlog

- [ ] Ottimizzare digitalizzazione testo passando il batch di bbox (extractor)
- [ ] Parallelizzare extractor
- [x] Possibile deadline
- [ ] Docker con cli d2, dipendenze python 
- [ ] Finetunare la rete di text digitization
- [ ] GUI


# Programma

Dataset: 
- Data augmentation: 2 giorni

Classifier:

**Fil**:
- Adeguare pipeline: 8 ore
- Preprocessing: 
  - Padding bianco centrato e fissare una dimensione dell'immagini: (sotto-metodo di preprocessing) 4 ore 
  - Rimozione quadretti con filtro mediano: (sotto-metodo di preprocessing) 4 ore 
  - Binarizzazione bianco nero (Otsu): (sotto-metodo di preprocessing) 4 ore 
  - Geometrizzazione per riallineare le immagini: (sotto-metodo di preprocessing) 8 ore 
- Modello: 4 giorni
- Test: 4 giorni


**Sav**:
Object detection:

- Preprocessing: 2 giorni
- Modello: 4 giorni
- Test: 4 giorni


**Nic**:
Frecce (trovare direzione, testa, coda):

- Costruzione Downsampling+upsampling network per predire una matrice di valori (+ softmax) per sapere dove è la testa e dove è la coda
- Test: 4+4 giorni
- Costruzione classificatore testa/no testa + sliding window
- Test: 4+4


- Fine tuning finale + cose a contorno: 5 giorni

----
356 ore
15gg*uomo per finire il progetto
4gg per fare il paper
2gg slide




