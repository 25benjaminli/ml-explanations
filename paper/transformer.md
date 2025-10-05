## How transformers work at a high level:

### First, with vectors. 

First, we generate embeddings (1 x d_e) for each token, say each has d_e= 2048 (representing depth of embedding). 

Next, these embeddings are multiplied by query (W_Q), key (W_K), value (W_V) matrices (each (d_e, d_model)) to create projections. 

Why have both query AND key, you may ask? It's because when you look at a word, it matters because interpreting language is NOT one-way. For instance, dog bites man is not the same as man bites dog. 

- these projections may be smaller than the embeddings since d_model can be as small as 256. 

The purpose of these embeddings is ??

To calculate self-attention, we need to score each word against other words in the sentence. 

Take the dot product of QUERY vector with KEY vector of the word we're scoring (yielding attention score)

Word 1 <-> Word 1 = Q1 * K1 (dot product)
Word 1 <-> Word 2 = Q1 * K2 (dot product)

The dot product works because the more different they are from one another, 
the smaller the attention score is. 

Next, divide by square root of d_model for more stable gradients. 
Pass all attention scores into softmax, normalizing all entries to sum up to 1. 

Next, multiply each VALUE vector by the softmax score. 

Finally, sum up the weighted value vectors, yielding the output of self attention for the first word. 

### Next, with matrices

Instead of having a vector of length d_e, we have a matrix X of dimensions (n, d_e) where n is the number of words currently being processed. 

Doing X * W_Q yields query matrix Q, which has dims (n, d_model) because Q is (d_e, d_model). Repeat for W_K and W_V to get K and V. 

Next, we calculate the attention scores. 

Instead of having to compare pairs of query and key matrices and taking the dot product, we can just multiply the two together. 

softmax(Q * K_T / sqrt(d_model)) * V

The inside expression evaluates to (n, n), representing how much each token at position i attends to that at position j. which is then multiplied by V (n, d_model) -> (n, d_model). Now, each row n_i in V encodes similarity information from other rows as well as itself. 

Multi head attention has multiple W_Q, W_K, W_V matrices. 

<!-- 

Ok, now we're onto the big leagues. Transformers are very different from FCNs or CNNs because of their \textbf{scale}. 

We are going to discuss transformers for NLP, but vision transformers have similar working principles. The general process by which transformers process data is as follows:

1. encode a group of words into vectors + combine into a matrix + add positional embeddings + apply batching if necessary (generating a vector n x embedding\_depth)

2. pass into multi-head attention mechanism, which will output a new matrix of the same dimensions (n x embedding\_depth)

3. pass into a multilayer perceptron

4. repeat steps 2 and 3 for a predefined number of times (transformer layers)

5. apply the same steps but backwards (this is the decoding part)

6. convert output matrix to relevant words from vocabulary (basically reversing the original embedding)


\hl{FULL DETAILS COMING SOON!}
 -->