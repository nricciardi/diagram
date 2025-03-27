Commenti nic:

I caratteri speciali come li gestiamo? e.g. caratteri matematici delle formule (molto probabili in diagrammi)

Non mi è chiarissimo cosa intendi per encoding del testo? Un vettore sparso? Forse troppo spendacciona come soluzione (?) 

(Te l'ho già detto, ma per completezza) con JSON io intendevo una classe (JSON è impropriamente usato per riferirsi a quello che poi sarebbe il dump)
che abbia al suo interno, per esempio, una o più liste di oggetti, invece che un singolo campo `content` di tipo tensore

(Sempre ammesso che io abbia capito), secondo me tenere un qualcosa come `elements` dell'esempio sotto ci può stare per tenersi traccia del testo
(e forse anche ulteriori campi per i bbox), poi le relazioni si potrebbe fare anche con il tensore. 
Secondo me è cruciale capire quante cose davvero si possono fare in GPU nel processo che va dall'uscita della rete dei bbox all'ottenimento
della stringa di output del trasduttore. Perché, per esempio, se l'overlapping non si può parallelizzare a dovere perdiamo molti dei vantaggi
di un approccio a tensore. Stessa cosa per il trasducer, se non si riesce a parallelizzare la creazione della stringa tocca sempre la CPU
(forse si può fare per diagrammi semplici, ma già con flowchart con giri strani ho dei dubbi)


e.g. (potrei non aver capito benissimo UDR, ma è solo un esempio per cosa intendo con "json")
```Python
class UnifiedDiagramRepresentation(DiagramRepresentation):
  
    elements: list
    # [
    # { id, type, label }
    # ]
    
    relations: list
    # [
    # { id_obj1, id_obj2, type }
    # ]
``` 

# Unified Representation 

| Tensor                          | JSON                             |
|---------------------------------|----------------------------------|
| **GPU**                         | CPU                              |
| **Natural format for networks** | Needs conversion by hand         |
| **Agnostic**                    | Needs some semantic injected (?) |
| Handles only numbers            | **Handles text**                 |
| Manageable only with Python     | **Manageable with everything**   |
| Less intuitive                  | **More human-readable**          |

Professor's suggestion: "if you can do everything with tensors, use tensors"

> [!CAUTION]
> Problem with tensor: text encoding 

> [!TIP]
> Possible solutions:
> 1. Numeric encoding of text 
>   - The result has to be an integer or a floating point 
>   - Difficult, allows to encode only single characters or small words 
> 2. A tensor and a dictionary
>   - Tensor to map the relations in the diagram
>   - Dictionary to keep track of the labels found in the diagram and their indices to associate them to the correct component
> 3. JSON
>   - To be defined if 1 and 2 fail

Why could 2. be a good trade-off? 
- We keep the advantages of using tensors for everything except text
  - Everything on GPU both in extractor and in transducer
- We add the advantages of JSON for managing text
- Easy to implement (we just need two attributes in the `UnifiedDiagramRepresentation` class)
- Graphic components (whose rendering does not depend on us) separated from text components (which have to be linked to some graphic components)
  - Since they have to be treated differently, having them isolated from each other sounds reasonable

Why could 3. be a good trade-off?
- Only one representation for everything 
- Easier to manage (and human-readable)
- Cleaner code

> [!IMPORTANT]
> Proposal for tensor (option 2.)
> Shape: (category, obj1, obj2) 
> - category is the type of the object (given by the annotations in the dataset)
>   - category is as long as the number of different annotations in the dataset
> - obj1 and obj2 are object IDs of two objects in the diagram (based on the output of the bounding box network in the extractor)
>   - obj1 and obj2 are as long as the number of different objects found by the network (+1 for the None object?)
> 
> Values: {0, 1}
>   - 0 means that there is no relation between the objects
>   - 1 means that there is a relation between the objects 
>   - **The kind of relation is known by the transducer, not the extractor**

> [!NOTE]
> If we use 2., since we have two different types of objects, (tensor and dictionary), two output files may be needed (`tensor.pt`, `labels.json`)

Possible (conceptual) implementation of 2.

```Python
class UnifiedDiagramRepresentation(DiagramRepresentation):
  
    relations: torch.Tensor # (category, obj1, obj2)
    labels: dict # {id: text}
    
    def __init__(self, relations: torch.Tensor, labels: dict):
        self.relations = relations
        self.labels = labels

    def dump(self, tensor_output_path: str, labels_output_path: str):
        torch.save(self.relations, tensor_output_path)
        json.dumps(self.labels, labels_output_path)

    def load(self, tensor_input_path: str, labels_input_path: str):
        relations = torch.load(tensor_input_path)
        labels = json.load(labels_input_path)
```

If we want to use 3. we have to define the file format

Possible (very conceptual) implementation of 3. 

```Python
class UnifiedDiagramRepresentation(DiagramRepresentation):
  
    diagram_representation: list # TODO define format
    
    def __init__(self, diagram_representation: list):
        self.diagram_representation = diagram_representation

    def dump(self, output_path: str):
        json.dumps(self.labels, output_path)

    def load(self, input_path: str):
        diagram_representation = json.load(input_path)
``` 


