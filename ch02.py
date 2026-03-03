# To install all the required packages run the following:
# pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt
import os
import re
import requests
import torch
#import importlib
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader
import tiktoken

#print('Main development environment for the support bot.')
if not os.path.exists("/teamspace/studios/this_studio/llm-from-scratch/the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "/teamspace/studios/this_studio/llm-from-scratch/the-verdict.txt"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)
with open("/teamspace/studios/this_studio/llm-from-scratch/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
#Converting the texts into tokens
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
[item.strip() for item in preprocessed if item.strip()]
#print("Total number of character:", len(raw_text))
#print(preprocessed[:30])
#print("Total tokens processed:", len(preprocessed))
#Converting the tokens into token IDs
all_words = sorted(set(preprocessed))
vocabulary_size = len(all_words)
#print("Vocabulary Size: ", vocabulary_size)
vocabulary = {token:integer for integer,token in enumerate(all_words)}

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

""" for i, item in enumerate(vocabulary.items()):
    print(item)
    if i >= 50:
        break """


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# Byte Pair Encoding (BPE)
#print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
""" text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integer = tokenizer.encode(text, allowed_special={"<|endoftext|>"}) # We add end of text token since GPT2 does not support by default
print(integer)

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:] # we are truncating the first 50 tokens
context_size = 4
x = enc_sample[:context_size] #first four tokens
y = enc_sample[1:context_size+1] #here we shift to the right by one position
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

print("PyTorch version:", torch.__version__) """

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                     stride=128, shuffle=True, drop_last=True,
                     num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")   
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

#dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
# Increase the stride to prevent overlapping and overfitting the LLM model
#dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=4, shuffle=False)
# Now we increase the batch_size to make it more efficient
#dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=4, shuffle=False)
""" data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch) """

# Now we increase the batch_size to make it more efficient
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
#print("Inputs:\n", inputs)
#print("\nTargets:\n", targets)

#input_ids = torch.tensor([2, 3, 5, 1])
#vocab_size = 6
#output_dim = 3

#torch.manual_seed(123)
#embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#print(embedding_layer.weight)
#This are the parameters we'll use to actually train the LLM
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
#print(token_embeddings.shape)
max_length = 4
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
#print(pos_embedding_layer.weight)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)
print(pos_embeddings)




