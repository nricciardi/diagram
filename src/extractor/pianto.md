
# Extractor

- freccia-nodi, 
- ci possono essere più testi associati a uno stesso componente (nodo/freccia)
- testo-nodo, testo dentro, testo fuori, più testi associati, i testi overlappati con il nodo 


## Pipeline

1. Preprocessing
2. Object detection -> nodi, frecce, testo
3. Filtro confidenza
4. [in parallelo]
   - Associare i testi alle frecce e ai nodi 
   - Digitalizzazione del testo
   - Overlapping nodi-frecce per relazioni (+ cercare testa e coda delle frecce)
5. [in parallelo]
   - Overlapping nodi-testo
   - Overlapping frecce-testo
6. Creazione DiagramRepresentation
7. Gestione del secchio degli scartati

```python

def extract(image):
   ...
    output = extract_bbox(image)
    overlap(output)

def extract_bbox(image) -> BboxOutput:
   
   model = ...
   output_rcnn = model(image)
   
   output: BboxOutput = transform(output_rcnn)
   
   return output

```
 



### Preprocessing

`_preprocess`

TODO

### Object detection -> nodi, frecce, testo

`_extract_diagram_objects`

TODO

### Filtro confidenza

TODO: scegliere la soglia (dipende da quanto brava è la rete)

### Associare i testi alle frecce e ai nodi

`_compute_text_associations`

Iterare sui bbox testo: 

Scegliere il bbox nodo/freccia più vicino (**senza soglia**)


### Digitalizzazione del testo

`_digitalize_text`

Da Chat:

   | Model | Library | Type | Size | Speed | Accuracy | Language Support | HuggingFace Hosted | Handwriting Support | Notes |
   | ------ | ------- | ------ | ------ | ------ | ------- | ------ | ------ |  ------ | ------- |
   | microsoft/trocr-small | Transformer (TrOCR) | ~100 MB | Medium | High (printed) | English (mostly) | ✅ Yes | ❌ (use -handwritten version) | Great for clean, printed text
microsoft/trocr-small-handwritten | Transformer (TrOCR) | ~100 MB | Medium | High (handwriting) | English | ✅ Yes | ✅ Yes | Optimized for handwriting
keras-io/ocr-crnn | CNN + RNN | ~30 MB | Fast | Medium-High | Latin-based only | ✅ Yes | ❌ | Lightweight, but needs TF setup
easyocr | CNN + RNN + CTC | ~60 MB | Fast | Good | 80+ languages | ❌ No (pip only) | ✅ Yes | Easiest to use, great multilingual support

Per ora provo con trocr-small-handwritten

TODO: chiedere se possiamo prendere una rete già fatta per il testo

### Overlapping nodi-frecce per relazioni

`_compute_relations`

freccia massimo due nodi, soglia per rimuovere frecce troppo lontane (e.g. dei None)

Scorriamo sulle frecce:

1. Cercare testa e coda delle frecce

2. Fillare "l'oggetto intermedio"
   - Per la testa prendiamo **il nodo più vicino**
   - Per la coda prendiamo **il nodo più vicino**

"più vicino": il più vicino tra i nodi con distanza **sotto-soglia**, altrimenti `None`
"overlap tra nodo e freccia" = distanza 0

**Considerare testa e coda**

=> per ogni freccia (source, target); source, target in (None, Node)

category data dalla bbox ("label" della freccia)

Riusciamo a riempire:

```python
category: str
source_id: int | None
target_id: int | None
```


### Overlapping nodi-testo

`_element_text_type`

**Per ogni testo** (=> associazione univoca testo-nodo):

```python
inner_text: List[str]
outer_text: List[str]
```

1. `inner_text` overlapping in percentuale tra bbox del testo e bbox del nodo (o viceversa) con una soglia **molta alta** (e.g. >= t=85%)
2. `outer_text`
    - non overlappa e ha una distanza minore di una soglia => `outer_text`; altrimenti nel **secchio degli scartati**
    - overlapping in percentuale tra bbox del testo e bbox del nodo (o viceversa) (e.g. x di overlapping tra i due) con una soglia **molta bassa** (e.g. se x < t=85% => `outer_text`)


### Overlapping frecce-testo

`_arrow_text_type`

```python
inner_text: List[str]
source_text: List[str]
target_text: List[str]
middle_text: List[str]
```

1. inner_text: bbox del testo che viene "completamente tagliata" dalla freccia 

     |------|
-----|-text-|--------
     |------|

2. Data la freccia*, la dividiamo in 3**, associamo al testo l'argmax tra l'overlapping delle zone oppure distanza sotto soglia (oppure **secchio degli scartati**)

----|--------------|-----




*"Vettorizzazione" del bbox; problema con frecce diagonali/curve... calcolare quanti pixel "neri" ci sono in quale diagonale?


** middle è più grande di source e target, attenzione alle frecce circoli e diagonali


### Creazione DiagramRepresentation

Durante il join dei processi (multiprocess) costruiamo la rappresentazione.


### Gestione del secchio degli scartati

TODO



