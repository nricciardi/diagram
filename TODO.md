# TODO

## Lunedì 31/03

- [x] ~~Rappresentazione universale~~
- [ ] Definire una nuova tipologia di diagramma
- [x] Trasduttori Mermaid e PlantUML per graph diagram e flow chart
- [x] Orchestratore sequenziale

## Lunedì 07/04

- [ ] Scelgiere un nuovo diagramma
- [ ] ~~fil: Refactor (non necessario) transducer (usare le librerie sul gruppo)~~
- [x] sav: Wrappare compilatore Mermaid (https://mermaid.js.org/ecosystem/tutorials.html#jupyter-python-integration-with-mermaid-js)
- [x] sav: Wrappare compilatore D2
- [x] nic: Multiprocess orchestratore
- [x] fil: Proposal


## Lunedì 14/04

- [ ] Dottorando: scelgiere un nuovo diagramma
- [ ] nic: CLI per main
- [ ] fil: Rollback transducer
- [x] sav: Definire l'output/input della rete dei bbox dell'extractor (07/04/2025 alle 10)
- [ ] Dataset (`DataLoader`)
  - [ ] Portare proposte
  - [ ] Parse del dataset per i nostri scopi
  - [ ] Fillare la classe wrapper `Image` per avere sia il "content opencv" sia annotazioni "json" come attributi 
  - [ ] Creare classe per gestire il dataset (e.g. get_image); guardare se c'è la classe 
- [ ] fil: Test completo transducer + compiler
- [ ] Classifier
  - [ ] Preprocessing
    - [ ] Definire -> chiedendo al dottorando
    - [ ] Togliere rumore (gaussian filter) [filter]
    - [ ] Togliere il background (e.g. togliere quadretti) [e.g. Otsu]
    - [ ] Raddrizzare immagini [geom]
  - [ ] Rete classificatrice
    - [ ] Definire -> chiedendo al dottorando
- Chiedere al dottorando per i bounding box
- Componente per overlapping (relazioni tra elementi dopo bounding)
  - [ ] sav: Definire come fare
  - [ ] Farlo

## Lunedì 21/04


