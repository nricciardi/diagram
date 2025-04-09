# Classifier

## Preprocessing

- Raddrizzare immagini [geom]
- Togliere rumore (gaussian filter) [filter]
- Togliere il background (e.g. togliere quadretti) [e.g. Otsu]



## Classificazione vera (TODO: saverio troverà un nome migliore)

TODO: al ~~dottorando~~ (ChatGPT)

Possibilità:

- **Handcrafted** e.g. KNN (numero elementi, ...) -> forse abbiamo qualcosa già fatto per l'extractor
- **Inferite** e.g. CNN classificatrice (estrae in autonomia le features)






## Considerazioni

Il testo è (quasi, e.g. rettangolo del class diagram con testo) irrilevante nella classificazione.

Il colore è irrilevante. Anche se esistessero diagrammi in cui il colore ha un significato, 
comunque è improbabile che ci siano due tipologie di diagrammi con le stesse figure ma con colori diversi.

Le sole forme, il numero di forme e la dimensione delle forme non è sufficiente per distinguere due tipologie di diagrammi.
Si consideri per esempio class diagram e flow chart, entrambi hanno frecce e rettangoli.
Bisogna considerare le forme e le relazioni tra gli oggetti, senza la necessità di comprenderne il significato (che verrà fatto nell'estrattore).

Nelle immagini è presente rumore come ad esempio i quadretti del foglio, sbavature, immagini inclinate.


## Resoconto

### Conversione in scala di grigi

Ridurre la complessità computazionale eliminando le informazioni di colore superflue.

https://www.sciencedirect.com/science/article/pii/S2667305324000346


### Riduzione del Rumore e Filtraggio

Rimuovere rumori, in teoria anche i quadretti del foglio con **filtro mediano** o **gaussiano**.

https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/ipr2.13243

### Binarizzazione e thresholding adattivo

Semplificare la segmentazione e l’estrazione di caratteristiche

- Thresholding di Otsu
- Adaptive Mean / Gaussian Thresholding

https://www.sciencedirect.com/science/article/pii/S2667305324000346


### Allineamento

Molti diagrammi sono scritti a mano in maniera inclinata.

Trasformata di Hough per rilevare l’angolo dominante e ruotare l’immagine di conseguenza.

Arrow R-CNN (Julca-Aguilar & Hirata, 2020)


### Ridimensionamento

Ridimensionare tutte le immagini a 224x224 px (standard per CNN come ResNet)
oppure tenere una nostra dimensione.

Consigliano di normalizzare i valori dei pixel tra 0 e 1 con media 0 e varianza 1.



### Data augmentation

Ridurre l’overfitting e migliorare la generalizzazione.

Si possono usare i transformer.

Handwritten Digit Recognition Using Deep Learning Algorithms, arXiv (2021)
https://arxiv.org/pdf/2106.12614



















