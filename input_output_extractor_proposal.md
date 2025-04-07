# Input/output extractor proposal 

The extractor consists of two components:
1. A bbox network 
2. An overlapping component whose objective is to find the relations between the elements of the diagram (e.g., for 
the graph diagram, which nodes are connected by an arrow)

## Input/output bbox net 

`input: Image` where 
```Python
class ImageConcreteImpl(Image):
    content: tensor # (iC, H, W) -> iC = 3 for RGB
    def __init__(self, file_path: str):
        ... #some openCV function probably
```

TODO:
- [ ] fix `H` and `W`

```Python
output: tensor # (category_id, confidence, x1, y1, x2, y2)
# dim(category_id) = dim(elements_in_the_diagram)
# dim(x1) = dim(y1) = dim(x2) = dim(y2) = dim(category)
```

The above is the **raw output of the bbox net**. Before feeding it to the overlapping component, we need to process it
(e.g., apply a threshold on the confidence)
- we may have different thresholds for different categories (e.g., nodes, arrows, labels)

TODO:
- [ ] Find the best threshold(s)

## Input/output overlapping component 

```Python
input: tensor # (category_id, x1, y1, x2, y2)
# obtained by applying a threshold on the raw output of the bbox net
output: DiagramRepresentation # for instance, the flowchart representation -> already defined
```

TODO
- [ ] Define overlapping algorithm more in detail