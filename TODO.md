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

- [ ] Preprocessing extractor [sgherro]
- [ ] Object detection -> nodi, frecce, testo [sgherro]
- [ ] Classifier
  - [ ] Preprocessing [sgherro]
    - [ ] Definire quali fare -> chiedendo al dottorando
    - [ ] Togliere rumore (gaussian filter) [filter]
  - [ ] Classifier: Raddrizzare immagini [geom]
  - [ ] Togliere il background (e.g. togliere quadretti) [e.g. Otsu]
  - [ ] Rete classificatrice
    - [ ] Definire -> chiedendo al dottorando



## Backlog

- [ ] Ottimizzare digitalizzazione testo passando il batch di bbox (extractor)
- [ ] Parallelizzare extractor
- [ ] Possibile deadline
- [ ] Docker con cli d2, dipendenze python 
- [ ] Finetunare la rete di text digitization
