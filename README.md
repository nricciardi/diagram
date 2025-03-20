# Progetto rita

[Scrivo in italiano per fare prima, poi cambiamo in inglese]

## Idea Diagrammi

TODO: portare alla rita degli esempi di diagramma, sta attendo prima di darci l'OK

- PlantUML
- Mermaid
- D2lang

Bisogna capire se si riesce a fare una cosa del tipo "scrivimelo nel linguaggio che più si adatta" oppure dobbiamo fissarlo a priori (e.g. train su Mermaid).

**EDIT:**

Repo con dataset diagrammi: https://github.com/bernhardschaefer/handwritten-diagram-datasets

Flow dell'applicazione:

1. Prende un'immagine di input
2. Tramite un classificatore, viene restituito il **tipo** di diagramma
3. Dato il tipo di diagramma, viene mandato in input alla rete **associata al tipo**
4. Ogni tipologia di rete restituisce un output "agnostico" rispetto al linguaggio da compilare
5. Ogni linguaggio compilabile ha il suo "parser" da output della rete a input per il compilatore (e.g. Mermaid)

_Esempio diagramma dei grafi:_

La rete è di fatto una objects detection che restituisce la doppia coppia di coordinate del box e la tipologia di oggetto (nodo, freccia, annotazione, ...).

Dopodiché, post processing per cercare tutti gli overlap tra box di nodi e box di frecce + overlap tra box di frecce e box delle annotazioni.

=> si ottiene quindi una rappresentazione _logica_ del grafo (e.g. una matrice del grafo stesso).

Il parser, ad hoc per ogni linguaggio (e.g. Mermaid), si occupa di costruire il codice effettivo da far compilare dal compilatore dei diagrammi.
