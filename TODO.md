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
- [ ] fil: Creazione DiagramRepresentation
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

- [ ] Sistemare file md sparsi in una documentazione più organica
- [ ] Parallelizzare extractor
- [ ] Cercare testa e coda delle frecce (rete) [sgherro]
- [ ] Preprocessing extractor [sgherro]
- [ ] Object detection -> nodi, frecce, testo [sgherro]
- [ ] Classifier
  - [ ] Preprocessing [sgherro]
    - [ ] Definire quali fare -> chiedendo al dottorando
    - [ ] Togliere rumore (gaussian filter) [filter]
    - [ ] Togliere il background (e.g. togliere quadretti) [e.g. Otsu]
  - [ ] Rete classificatrice
    - [ ] Definire -> chiedendo al dottorando

## Lunedì 05/05

- [ ] Classifier: Raddrizzare immagini [geom]
- [ ] Docker con cli d2, dipendenze python 
- [ ] Dottorando: scegliere un nuovo diagramma


## Venerdì 27/06

Possibile deadline


