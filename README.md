# Graph Convolutional Networks From Scratch
My attempt at building the gears of a graph convolutional network (GCN) from the ground up.

Through the process, [Jonathan Hui's blog post]([url](https://jonathan-hui.medium.com/graph-convolutional-networks-gcn-pooling-839184205692)) and [Inneke Mayachita's explainer]([url](https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b)) were the gold-standards from which I learned.

**This is not meant to be a zero-to-one primer. This is moreso my notes on GCNs, condensed into an explainer for my own comprehension (Feynman Technique).**

## Components of a minimal(ish) GCN
Just like the analogous convolutional networks, there are a few core classes and operations.

### 1. Adjacency Matrix

Graph networks require a graph structure as input. Adjacency matrices are convenient ways to represent graph information. Cells with a '1' represent connections between nodes, '0' means there's no connection. In the image, Node 0 shares an edge with Node 1, 2, and 3.

<img width="300" alt="Screenshot 2024-03-30 at 10 46 55 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/5f5feb62-ac3d-4bc7-8775-29e0c0821a47">

Edge indices are also convenient representations of matrices. Edge indices are a list of node-node connections, e.g. [[1,0], [2,0], [2,3]] is a graph where Node 0 shares an edge with Node 1 and 2, and Node 2 and 3 also share an edge.

Edge indices are often more efficient than a full adjacency matrix (which grows at a complexity of N^2, where N is the number of nodes).

However, in this case I chose to use the matrix representation because there were only 34 nodes in the graph, and it simplified much of the calculations.

### 2. Feature Matrix

Just as NLP involves tokens with word embeddings, each node in a graph network has its own feature vector. (So does each edge, but we'll ignore edges for now).

The feature matrix is simply an matrix of every node feature vector, and as a result it is an N x feature_dim matrix, where N is the number of nodes and feature_dim is the number of features of your embedding vector.

This carries information about each node. More on this in a bit.

### 3. Self-Loops

Naively, you might could multiply the adjacency matrix (A) and feature matrix (X) to somehow combine information from the graph connections and the information present in each node.

What happens when you perform the matmul? Well...

<img width="400" alt="Screenshot 2024-03-30 at 10 52 38 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/ba452e58-1329-41be-b4e1-0cd6880ce685">

AX turns out to sum all the neighboring feature vectors for each node! e.g. AX[0] = 1 + 2 + 3. This is pretty good if our goal is to aggregate information from the graph. But there's a problem. If we go back to the adjacency matrix, you'll notice that all the diagonals are 0s.

<img width="400" alt="Screenshot 2024-03-30 at 10 55 12 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/075b6492-5c81-45d3-a284-ac7095e66e69">

**For each node, AX doesn't take into account the features of node itself.** We can resolve this with self-loops, essentially making it so that each node connect to itself.

<img width="400" alt="Screenshot 2024-03-30 at 10 58 39 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/88ecfd4e-57ed-4b61-90d9-444061ca1c26">

In the adjacency matrix, we can simply add 1s along the diagonal by performing a simple sum: A' = A + I. The resulting matrix and product is:

<img width="400" alt="Screenshot 2024-03-30 at 10 51 59 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/10d5e67e-835b-4015-935c-faacf018655a">

You can see that now AX takes into account information of each node itself when performing the sum.

### 4. Normalized Adjacency Matrix

There's another more subtle problem with the current implementation. If you look at the product of AX, **they aren't normalized.** What does this mean? Well, if we keep applying the same operation to the graph, we'll encounter vanishing/exploding gradients. Similar to regular neural networks, we'll need to normalize to prevent numerical instabilities.

We can normalize our GCN by finding the Degree Matrix (D) and multiplying the inverse of D with our adjacency matrix (A).

<img width="400" alt="Screenshot 2024-03-30 at 11 07 02 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/f516aa5a-d325-4863-a567-582a411577f3">

Comparing against the same mat mul as before, you can see that now the product is normalized.

<img width="400" alt="Screenshot 2024-03-30 at 11 07 19 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/cf44323b-1e4c-4627-833b-4de9641392f2">

Without diving into too much of the math (you can do that [here]([url](https://arxiv.org/pdf/1609.02907.pdf))), Kipf and Welling showed that the ideal normalization actually takes on a symmetric form.

<img width="456" alt="Screenshot 2024-03-30 at 11 08 12 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/492abb4e-c1d2-4f23-9ed3-845ca9943208">

Now the output is:

<img width="400" alt="Screenshot 2024-03-30 at 11 09 34 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/f100bde5-4b7d-4e6a-9a64-625ca8aa6d9a">

That's all for normalization.

### 5. Convolution / Propagation

Putting everything together, performing the single layer of convolution across the entire graph takes the form of:

<img width="400" alt="Screenshot 2024-03-30 at 11 11 56 PM" src="https://github.com/linjames0/gnn-from-scratch/assets/78285353/94b43b64-ae24-42c7-a463-3604d1b1179f">

Where D is the degree matrix, A is the adjacency matrix with self-loops, H^l is feature matrix at layer l (H^0 is just X, the initial feature matrix), and W^l is the weight matrix of the neural network layer that processes the result of DADX.

## Conclusion

That's all for now. Thanks for reading if you made it this far! If you want to learn more, I highly recommend this [Graph Neural Network Primer]([url](https://distill.pub/2021/gnn-intro/)) in addition to both posts I mentioned above ([Jonathan]([url](https://jonathan-hui.medium.com/graph-convolutional-networks-gcn-pooling-839184205692)), [Inneke]([url](https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b)))
