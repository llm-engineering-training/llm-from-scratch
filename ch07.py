from importlib.metadata import version
import json
import os
import requests
import torch
import tiktoken
from torch.utils.data import Dataset
from functools import partial
from torch.utils.data import DataLoader
from gpt_download import download_and_load_gpt2
from ch04 import GPTModel
import time
from tqdm import tqdm
import re
import psutil
from ollama import Client
import ollama

from ch05 import (
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
    generate,
    calc_loss_loader,
    train_model_simple,
    plot_losses
)

pkgs = [
    "numpy",       # PyTorch & TensorFlow dependency
    "matplotlib",  # Plotting library
    "tiktoken",    # Tokenizer
    "torch",       # Deep learning library
    "tqdm",        # Progress bar
    "tensorflow",  # For OpenAI's pretrained weights
]

tokenizer = tiktoken.get_encoding("gpt2")

""" for p in pkgs:
    print(f"{p} version: {version(p)}") """

""" 
7.2 Preparing a dataset for supervised instruction finetuning
format used in instruction-data.json is called the Alpaca prompt
style template
But it is not what is inputed to the LLM.
In general the Apaca format template will be formatted as

SOME APPENDED GENERAL INSTUCTION PROMPT
### Instruction
Identify the correct spelling of the following word
### Input
Ocassion

### Response
The correct spelling is 'Occasion'

Microsoft Phi-3 prompt style template
<|user|>
Identify the correct spelling of the following word 'ocasion'

<|assistant|>
The correct spelling is 'Occasion'


 """

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
#print("Number of entries:", len(data))

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

""" model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"

print(model_input + desired_response) """

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

""" 
print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))
 """

""" 
7.3 Organizing data into training batches
This involves the following steps
1 - format data using prompt template
2 - Tokenize the formatted data
3-  Adjust to the same length with padding tokens
4 - Create target token IDs for training
5 - Replace padding tokens with placeholders


 """

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)

#print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    """ Find the longest sequence in the batch
    and increase the max length by 1, which will add one extra
    padding token below
         """
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []
    
    for item in batch:
        new_item = item.copy()
        """ Add and <|endoftext|> token
         """
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        """  Via padded[:-1] we remove the extra padded token
        that has been added via the +1 setting in the batch_max_length
        The extra padding token will be relevant in later codes
         """        
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor



def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    """ Find the longest sequence in the batch
    and increase the max length by 1, which will add one extra
    padding token below
         """
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        new_item = item.copy()
        """ Add and <|endoftext|> token
         """
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        """  Via padded[:-1] we remove the extra padded token
        that has been added via the +1 setting in the batch_max_length
        The extra padding token will be relevant in later codes
         """        
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_lst

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

""" 
Use the examples below to see what the tensor output would like
from a call to custom_collate_draft_1
cross entropy loss will ignore values of -100
 """
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

#print(custom_collate_draft_1(batch))
#inputs, targets = custom_collate_draft_2(batch)
""" inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets) """

""" 
7.4 Creating data loaders for an instruction dataset


 """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)
num_workers = 0
batch_size = 8

torch.manual_seed(123)
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

""" print("Train loader:")
for inputs, targets in train_loader:
    pass
    #print(inputs.shape, targets.shape)
print(inputs[0])
print(targets[0]) """


""" 
7.5 Loading a pretrained LLM


 """
BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

""" input_text = format_input(val_data[0])

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = (
    generated_text[len(input_text):]
    .replace("### Response:", "")
    .strip()
)
print(input_text)
print(response_text) """




""" 
7.6 Finetuning the LLM on instruction data
The data used here is refered to as instruction fine tuning

 """
model.to(device)
""" with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print("Training loss:", train_loss)
print("Validation loss:", val_loss) """

""" 
Uncomment below this to train the model

start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

End model training

 """

""" 
7.7 Extracting and saving responses
The data used here is refered to as instruction fine tuning

 """

""" 
Uncomment the code below to generate model responses and save them to a json file 
and also saved the newly trained weights

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text


with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

print(test_data[0])
#Here we save the trained model 
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")
# Load model via
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

 """

""" 
To continue evaluating the model responses generated from the code above we are going to use ollama

curl -fsSL https://ollama.com/install.sh > install.sh
On line 162 we replaced 
OLLAMA_INSTALL_DIR=$(dirname ${BINDIR})
with
OLLAMA_INSTALL_DIR="/teamspace/studios/this_studio/llm-from-scratch"
chmod +x install.sh
./install.sh 
export PATH=$PATH:/teamspace/studios/this_studio/llm-from-scratch/bin
ollama serve
Then in a second terminal window run ollama run <model_name> after running the export command
ollama run llama3

llama_server: server is listening on http://127.0.0.1:51227

ollama pull llama3
pip install ollama
To end ollama session type /bye
 """

ollama_url="http://127.0.0.1:11434"

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
        return running

#print("Ollama running:", check_if_running("ollama"))
def check_if_running_2(process_url):
    try:
        response = requests.get(process_url)
        if response.status_code == 200:
            print("Ollama is running")
    except requests.ConnectionError:
        print("Ollama is not running")

""" 
check_if_running_2(ollama_url)
 """

file_path = "instruction-data-with-response.json"

with open(file_path, "r") as file:
    test_data = json.load(file)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

def query_model(
    prompt,
    model="llama3",
    url=ollama_url
):
    data ={
        "model":model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            response_json = json.loads(line)
            if "message" in response_json:
                response_data += response_json["message"]["content"]

    return response_data


""" 

You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.
You will be given an instruction, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing the evaluation criteria.
Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
Please do not generate any other opening, closing, and explanations.

Here is the rubric you should use to build your answer:
1: The response fails to address the instructions, providing irrelevant, incorrect, or excessively verbose information that detracts from the user's request.
2: The response partially addresses the instructions but includes significant inaccuracies, irrelevant details, or excessive elaboration that detracts from the main task.
3: The response follows the instructions with some minor inaccuracies or omissions. It is generally relevant and clear, but may include some unnecessary details or could be more concise.
4: The response adheres to the instructions, offering clear, accurate, and relevant information in a concise manner, with only occasional, minor instances of excessive detail or slight lack of clarity.
5: The response fully adheres to the instructions, providing a clear, accurate, and relevant answer in a concise and efficient manner. It addresses all aspects of the request without unnecessary details or elaboration

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the instruction, the reference answer, and the response.

Instruction: {instruction}
Reference Answer: {reference}
Answer: {answer}

 """






""" 
# Connect to the local server

client = Client()
response = client.chat(
    model='llama3',
    messages=[
        {'role': 'user', 'content': 'Why is the sky blue?'}
    ]
)
print(response['message']['content'])
 """

""" 
def query_local_model(prompt_text):
    # Call the local Ollama API
    response = ollama.generate(
        model='llama3.2', 
        prompt=prompt_text
    )
    
    # Return the generated text
    return response['response']

if __name__ == "__main__":
    query = "Explain how Retrieval-Augmented Generation (RAG) works in simple terms."
    print(f"Query: {query}\n")
    
    answer = query_local_model(query)
    print(f"Answer:\n{answer}")
 """



