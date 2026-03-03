from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from importlib.metadata import version

# This is all part of the prerequisite
# Sequential class that allows us to stack layers and operations in order great for simple
# Linear is a fully  is a fundamental module in PyTorch that applies an affine linear transformation to the incoming data, 
# commonly known as a fully connected layer in neural networks
# in_features specifies the size of each input sample
# out_features specifies the size of each output sample resulting from the linear transformation
model = nn.Sequential(
    nn.Linear(in_features=3, out_features=2, bias=True), 
    nn.ReLU(),

    nn.Linear(in_features=2, out_features=4, bias=True), 
    nn.ReLU(),

    nn.Linear(in_features=4, out_features=2, bias=True), 
    nn.Softmax(dim=1),
)
sample_input = torch.randn(2,3)

#print("Sample Input:\n", sample_input)

output = model(sample_input)

#print("Output:\n", output)

""" # Dot product with numpy
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a,b) # also done with a@b
print("Dot Product:\n", dot) """

""" # Dot product with pytorch
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
dot = torch.dot(a,b) # also done with a@b. dot operator expects 1-dimensional vecors
print("Dot Product:\n", dot) """

# Matrix Multiplication with numpy
""" A = np.array([[1, 2],
        [3, 4]])
B = np.array([[5, 6],
        [7, 8]])
mult = A@B # same result with np.matmul(A,B)
print("Matrix Multiplication:\n", mult) """

#Now we look at matrix mult in ML scale
""" X = torch.randn(100,128)
W = torch.randn(128,256)
Y = X @ W
print("ML Scale Shape:\n", Y.shape) """
#Below we are looking at a simple buffer
class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        # Register a buffer called "scale"
        self.register_buffer("scale", torch.tensor(0.5))

    def forward(self, x):
        return self.linear(x) * self.scale 
# Each row below is an embedding vector
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
input_query = inputs[1]  # 2nd input token is the query
#print("input_query:\n", input_query)
""" res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * input_query[idx]

print(res) """
#print(torch.dot(inputs[0], input_query))

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, input_query) # dot product (transpose not necessary here since they are 1-dim vectors)

#print(attn_scores_2) # these are the scores...measure of similarities using the dot product
# In essence the above is taking the dot product of one input to measure similartity to the other inputs

""" attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum()) """

# Attention weights with softmax function - not recommended in practice
#Created to show how things work under the hood
""" def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0) """
#attn_weights_2_naive = softmax_naive(attn_scores_2)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

#print("Attention weights:", attn_weights_2)
#$print("Sum:", attn_weights_2.sum())

#Now we need to compute the context vector
context_vec_2 = torch.zeros(input_query.shape)
for i,x_i in enumerate(inputs):
    #print(f"{attn_weights_2[i]} --> {inputs[i]}")
    context_vec_2 += attn_weights_2[i]*x_i

#print(context_vec_2) #this is the context vector for the 2nd input with respect to the entire input matrix

#Starting 3.3.2 Simple self-attention mechanism (without trainable weights)
attn_scores = torch.empty(6, 6)
""" for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j) """#Naive method
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=1)
# print("All Attn Weights:", attn_weights)
all_context_vecs = attn_weights @ inputs
# print("All Context Vectors:", all_context_vecs)
#print("Inputs matrix:", inputs)

#Section 3.3 We compute the context vector with respect to the second input
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
# Create the following radomly initialized matrices
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value
keys = inputs @ W_key 
values = inputs @ W_value
#print("keys.shape:", keys.shape)
#print("values.shape:", values.shape)
#print("query_2 matrix:", query_2)
#keys_2 = keys[1]
#attn_score_22 = query_2.dot(keys_2)
# In Python we compute the square-root using notation d_k**0.5
attn_scores_2 = query_2 @ keys.T
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
context_vec_2 = attn_weights_2 @ values
#print(context_vec_2)
#No we generalize the above computation to the entire inputs matrix

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
#print(sa_v1(inputs))

# *3.5.1 Hiding future words with causal attention mask
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
#print("attention_weights", attn_weights)
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
#print("Simple Mask", mask_simple)
masked_simple = attn_weights*mask_simple
#print("Masked weights\n", masked_simple)
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
#print("Masked weights-Normalized\n", masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
#print("New masked", masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
#print("Masked weights", attn_weights)

# 3.5.2 Masking additional attention weights with dropout
dropout = torch.nn.Dropout(0.5) # dropout rate of 50%
example = torch.ones(6, 6) # create a matrix of ones
#print(dropout(attn_weights))

#3.5.3 Implementing a compact causal self-attention class

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forward method. 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec
batch = torch.stack((inputs, inputs), dim=0)
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
#context_vecs = ca(batch)
#print(context_vecs)
#print("context_vecs.shape:", context_vecs.shape)

# 3.6.1 Stacking multiple single-head attention layers
#Usually there is more than 2 heads - it is arbitrary
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

""" context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
) """
#context_vecs = mha(batch)
#print(context_vecs)
#print("context_vecs.shape:", context_vecs.shape)

#3.6.2 Implementing multi-head attention with weight splits
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
        # this will result in errors in the mask creation further below. 
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forward method.

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)






