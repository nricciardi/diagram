# Overlapping Algorithm

# Pippo's Proposal 

## Bins

Given $O$ object with properties $[x_1, y_1, x_2, y_2, \text{class}]$, we will put every object detected into a bin of index $i$. An object $O$ gets into $b_i$ - so, $O \in b_i$ - iff:

$$
\forall o \in b_i,\;\text{isOverlap}(o, O)
$$

The $\text{isOverlap}(obj_1, obj_2)$ returns $\text{True}$ if the two objects' coordinates overlap.

At first, a bin for every object will be created; then, until convergence, every possible couple of bins will be merged if possible.

## Bins Checking

A bin $b_i$ will be said to be valid iff:

$$
1 + \sum_{O\,\in\,b_i}{\mathbb{1}_{\{O.\text{class = relation}\}}} = \sum_{O\,\in\,b_i}{1_{\{O.\text{class = element}\}}}
$$

Since every relation links two different objects.

If a certain bin $b_i$ is found to be invalid, then there should be - in the working hypothesis that the bbox net works well - *another* invalid bin $b_j$ that should be merged with $b_i$.

Denoting:

- $B_I = \{i\,|\,b_i\;\text{is invalid}\}$

Pseudocode for merging:

1. While $\#B_I > 1$:
   
   1. Find $o_1, o_2 = \argmin_{i,\,j}{||o_i - o_j||^2}$ with $o_1 \in b_a,\,o_2 \in b_b$ with $b_a \neq b_b$.
   
   2. Merge $b_a,\, b_b$: $b_c = \{o\,|\,o \in b_a \lor\,o\in b_b\}$
   
   3. Redefine $B_I:=B_I\,-\{b_a,\,b_b\}\,\cup\{b_c\}$.

# Saverio's Proposal 

(Probably the same as Pippo's, but I'm not sure whether I fully understood it). 

## Hypotheses 

(0. Every object has at least one relation with another object)
1. If two objects overlap, they are in a relation
2. If an object does not overlap with anything, it is in relation to the nearest object to it (nearest = the one at 
the minimum distance)
3. Relations are reciprocal (e.g., A is in relation to B iff B is in relation to A)

## Idea 

Given the coordinates `[x1, y1, x2, y2]` of the `N` bbox detected `boxes: FloatTensor[N, 4]` we define the overlapping
function as 
```Python
def are_overlapped(bbox1: FloatTensor[4], bbox2: FloatTensor[4]) -> bool:
    # list of cases using the coordinates (trivial)
    # if at least a coordinate of one bbox is "contained" in the other one, there is overlap
```
and the distance function as 
```Python
def compute_distance(bbox1: FloatTensor[4], bbox2: FloatTensor[4]) -> float:
    # compute the pair-wise distance between all the vertices of the two bboxes and keep the minimum (trivial)
    # pair-wise distance = difference
```

## Algorithm 

The algorithm is the following

```Python
def find_relations(boxes: FloatTensor[N, 4], labels: Int64Tensor[N]) -> Tensor:
   relations: Tensor[R, 2]  # pair-wise relation tensor: every row has two objects, R is the number of relations found
   R = 0
   threshold = ...  # TODO define threshold
   for i, box1 in boxes:
      for j, box2 in boxes:
         if i == j:
            continue
         if are_overlapped(bbox1=box1, bbox2=box2):
            relations[R] = [i, j]
            R += 1
         else:
            distance = compute_distance(bbox1=box1, bbox2=box2)
            if distance < threshold:
               relations[R] = [i, j]
               R += 1
   return relations


def jsonify_relations(relations: Tensor[R, 2]) -> List[Relation]:
   json_relations: List[Relation]
   json_relation: Relation
   for relation in relations:
      json_relation.source_id = relation[0]
      json_relation.target_id = relation[1]
      json_relation.category = ...  # How do we get the category of the relation from the bbox net?
      json_relation.text = ...  # The "label" we are looking for here (which is different from the "labels" the bbox net
      # will return) are objects rather than attributes for the bbox: how do we turn an object into an attribute for 
      # another object? 
      json_relations.append(json_relation)

   return json_relations


def jsonify_elements(labels: Int64Tensor[N]) -> Dict[str, Element]:
   elements: Dict[str, Element]
   element: Element
   for i, label in labels:
      element.identifier = i
      element.category = label
      element.text = ...  # Same problem of json_relation.label 
      elements[i] = element

   return elements


def algorithm(boxes: FloatTensor[N, 4], labels: Int64Tensor[N]) -> DiagramRepresentation:
   elements: Dict[str, Element]
   relations: Tensor[R, 2]
   json_relations: List[Relation]
   representation: FlowchartRepresentation

   elements = jsonify_elements(labels=labels)
   relations = find_relations(boxes=boxes, labels=labels)
   json_relations = jsonify_relations(relations=relations)

   representation.elements = elements
   representation.relations = json_relations
   return representation
```











