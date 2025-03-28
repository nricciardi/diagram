Commenti nic:

I caratteri speciali come li gestiamo? e.g. caratteri matematici delle formule (molto probabili in diagrammi)
- We use a Python data structure to manage text, the text itself is given by the extractor (more specifically, by the bounding box)

Non mi è chiarissimo cosa intendi per encoding del testo? Un vettore sparso? Forse troppo spendacciona come soluzione (?) 
- Encoding text with a tensor is not a feasible nor an efficient solution, so we just eliminate it from the options

(Te l'ho già detto, ma per completezza) con JSON io intendevo una classe (JSON è impropriamente usato per riferirsi a quello che poi sarebbe il dump)
che abbia al suo interno, per esempio, una o più liste di oggetti, invece che un singolo campo `content` di tipo tensore
- Yes, we have to decide if use Python data structure + tensor or only Python data structures

(Sempre ammesso che io abbia capito), secondo me tenere un qualcosa come `elements` dell'esempio sotto ci può stare per tenersi traccia del testo
(e forse anche ulteriori campi per i bbox), poi le relazioni si potrebbe fare anche con il tensore. 
- Yes, I included it in the tensor + Python data structure conceptual implementation

Secondo me è cruciale capire quante cose davvero si possono fare in GPU nel processo che va dall'uscita della rete dei bbox all'ottenimento
della stringa di output del trasduttore. Perché, per esempio, se l'overlapping non si può parallelizzare a dovere perdiamo molti dei vantaggi
di un approccio a tensore. Stessa cosa per il trasducer, se non si riesce a parallelizzare la creazione della stringa tocca sempre la CPU
(forse si può fare per diagrammi semplici, ma già con flowchart con giri strani ho dei dubbi)
- Agreed, to be discussed more in depth

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

 Commenti Pippo:
1. Nel caso 2 - tensore con testo da tenere in un dizionario per intenderci - se i valori possiibli del tensore sono sempre 0 o 1, nel file che lega testo-oggetti c'è l'informazione degli oggetti a cui si lega?
exempli gratia:
In un GD c'è un nodo A (indice) con del testo all'interno, specificatamente "buongiorno". Si ottiene dunque:

```python
  tensor("testo", "A", "A") -> 1
            &
  {
    ["A", "A"]: "buongiorno"
  }
```
?

I was thinking about something more like (see the definition of `elements`)

```Python
tensor('node', 'A', 'A') -> 1 # the component A of the diagram (which is of type 'node') is in relation with itself
        & # Recall elements has the format [ {id, type, label} ]
{
  ['A', 'node', 'buongiorno'] # we can also see it as ['A', 'node']: 'buongiorno', it's a matter of representation
}
```

# Unified Representation 

## What we decided 

1. Python data structure to handle text 

## What we have to decide

1. Universal vs Specific
2. Tensor vs Python data structure for the relations

Possibilities
- Domain-specific structure
- Tensor of "pure" relations (for instance, arrows are concepts rather than components to be represented)
- Tensor of relations as links between components (arrows are components exactly like nodes etc)

### Universal vs Specific 

| Universal                                                                               | Specific |
|-----------------------------------------------------------------------------------------|----------|
| Sparse (has to model all the components for each diagram, even the ones it doesn't have | Dense (it only models the components of that specific diagram) |
| Same output of the extractor (and same input to all the transducers) for all diagrams   | Different output of the extractor (and different output to the transducers) depending on the type of diagram |
| "One" extractor | "Multiple" extractors (not necessarily one for each diagram) |

Universal:
- Higher memory requirements
- Easier implementation of the transducers (basically we have a template in which we only change the string to be passed to the compiler)
- Single-point-of-failure in the extractor (if it crashes, we are not able to process any diagram)
- Bottleneck in the extractor (if the extractor slows down, everything slows done)

Specific:
- Lower memory requirements
- Slightly harder implementation of the transducers (they are tailored to the specific diagram)
- Higher robustness (if one extractor crashes, we are not able to process only that family of diagrams)
- Better load-balancing (if we have different diagrams they will go to different extractors, reducing the risk of having a bottleneck)

Both:
- Able to reuse the previous knowledge when dealing with new diagrams 

### Tensor vs Python data structure (relations)

| Tensor                          | Python data structure                                                    |
|---------------------------------|--------------------------------------------------------------------------|
| **GPU**                         | CPU                                                                      |
| **Natural format for networks** | Needs conversion by hand                                                 |
| **Agnostic**                    | Needs some semantic injected                                             |
| Handles only numbers            | **Handles text**                                                         |
| Manageable only with Python     | **Manageable with everything** (we are considering "common" data structures) |
| Less intuitive                  | **More human-readable**                                                  |

Professor's suggestion: "if you can do everything with tensors, use tensors"
- Spoiler: we can't

Tensor:
- Useful if we can exploit the GPU power for most of the time 

Python data structure:
- If we spend most of the time on CPU, it may be easier to work with
- We can change the representation from numbers to text to be clearer 
  - We can also have the exact copy of the tensor representation (but without the GPU to use)

### Possibilities 

TODO (I will, I promise)


### Format Proposals (relations)

#### Tensor shape

> [!IMPORTANT]
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
> If we use tensor, since we have two different types of objects, (tensor and dictionary), two output files may be needed (`tensor.pt`, `labels.json`)

Possible (conceptual) implementation of tensor.

```Python
class UnifiedDiagramRepresentation(DiagramRepresentation):
  
    relations: torch.Tensor # (category, obj1, obj2)
    elements: list
    # [
    # { id, type, label }
    # ]
    
    def __init__(self, relations: torch.Tensor, elements: list):
        self.relations = relations
        self.elements = elements

    def dump(self, tensor_output_path: str, elements_output_path: str):
        torch.save(self.relations, tensor_output_path)
        json.dumps(self.labels, elements_output_path)

    def load(self, tensor_input_path: str, elements_input_path: str):
        self.relations = torch.load(tensor_input_path)
        self.elements = json.load(elements_input_path)
```

#### Python data structure

Possible (very conceptual) implementation of Python data structure 

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
    
    def __init__(self, elements: list, relations: list):
        self.elements = elements
        self.relations = relations

    def dump(self, output_path: str): # Not sure of how dumps works, it's just conceptual
        json.dumps(self.elements, output_path)
        json.dumps(self.relations, output_path)

    def load(self, input_path: str): # Not sure of how loads works, it's just conceptual
        self.elements = json.load(input_path)
        self.relations = json.load(input_path)
``` 


