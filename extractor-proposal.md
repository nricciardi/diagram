# Proposte extractor


## Preprocessing


### Fil





### Sav





### Nic

Per me ha senso provare lo stesso preprocessing del classificatore, 
magari con un thresholding meno aggressivo per evitare di perdere del testo o delle frecce.



## Object detection generale

**Object detection** (fine-tuning): trovare reti già fatte (possibilmente sui diagrammi o simili)
  - Dataset: FA, FAB, ...


### Fil





### Sav





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





### Sav





### Nic





