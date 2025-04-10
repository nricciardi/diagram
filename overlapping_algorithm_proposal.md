# Overlapping Algorithm

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
1 + \sum_{O\,\in\,b_i}{1\{O.\text{class = relation}\}} = \sum_{O\,\in\,b_i}{1\{O.\text{class = element}\} = \sum_{O\,\in\,b_i}{1\{O.\text{class = text}\}}}
$$


