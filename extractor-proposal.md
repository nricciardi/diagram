# Proposte extractor


## Preprocessing


### Fil

Forse qui si può evitare la binarizzazione... Non so. C'è il grosso problema del fixed size per la CNN,
a meno che non si spacchetti un'immagine in più immagini e le si mandino in batch.

Per le altre cose, direi comunque una normalizzazione dei pixel.


### Sav





### Nic

Per me ha senso provare lo stesso preprocessing del classificatore, 
magari con un thresholding meno aggressivo per evitare di perdere del testo o delle frecce.



## Object detection generale

**Object detection** (fine-tuning): trovare reti già fatte (possibilmente sui diagrammi o simili)
  - Dataset: FA, FAB, ...


### Fil

Ho cercato - ma non trovato - reti che riconoscessero semplici oggetti geometrici, forse perchè è un problema
risolto con la geometria.
Io tenterei con una CNN anche qui, perchè il dataset è piccolo.



### Sav

Open-source pre-trained networks:
- [detecron2](https://github.com/facebookresearch/detectron2.git)
  - Should integrate DETR



### Nic

Unica rete open che ho trovato che lavora su qualcosa di simile a diagrammi fatti a mano:
https://github.com/aaanthonyyy/CircuitNet

Articoli interessanti di cui non ho trovato la rete:

- https://link.springer.com/chapter/10.1007/978-3-030-86549-8_39
- https://link.springer.com/article/10.1007/s10032-020-00361-1

Reti pre-trained open:

- (non abbiamo avuto una bella esperienza ma) YOLOv8: https://github.com/ultralytics/ultralytics
- Faster R-CNN: supportato da torchvision https://pytorch.org/vision/main/models/faster_rcnn.html 
- DETR (o DINO): https://github.com/facebookresearch/detr; ma secondo chat "È necessario un numero consistente di dati per ottenere ottimi risultati.", non proprio il nostro caso



## Frecce

**Ottenere rappresentazione freccia** (as-is) (retta passante per test* o cod* da cui ottenere bbox 4 punti): provare [rete](https://link.springer.com/article/10.1007/s10032-020-00361-1) oppure trovarne un'altra (oppure piangere e farla noi)
Cercare testa e coda delle frecce (rete)



### Fil

Senza scendere nell'approccio rete, si potrebbe fare qualcosa di puramente geometrico.
Non so quanto sia fattibile, ma:
  1. Trovare il countour della freccia (insieme di punti del contorno) che con OpenCV possono essere collassati (si tengono solo i keypoints)
  2. Trovare la zona dell'immagine con un keypoint che ha un certo angolo dagli altri due (20°-60°) (se si hanno solo i punti 'focali', ma lo dice Chat)
  3. Dire che lì c'è la punta della freccia.

In codice:
```python
# Preprocessing
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

E' da rifinire e da tenere magari per far vedere alla Rita che non abbiamo cacciato reti a fiumi;
  non datemi delle sberle.

### Sav

#### CNN approach

Arrow R-CNN for handwritten diagram recognition

Cons:
- Apparently the network is not available

Pros:
- Seems to be the only one that suits our needs 
- We have the [paper](https://link.springer.com/article/10.1007/s10032-020-00361-1) 
  - we have the architecture
  - we have the datasets
  - we can "rebuild" and "re-train" the network from scratch (maybe we can use this as one of the "made-by-us" networks?)

#### Non-network approach

[Approach without network](https://stackoverflow.com/questions/66901546/finding-keypoints-in-an-handwritten-arrow)

Cons:
- Probably needs some adjustments

Pros:
- No CNN to train (maybe could be a disadvantage?)
- Few pre-packed opencv functions


### Nic





