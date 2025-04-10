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












