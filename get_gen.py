import torch
import torch.nn as nn
from datasets import Dataset
import types
import random
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,)
import numpy as np
import re
from datasets import load_dataset
import random
random.seed(2025)

parser = argparse.ArgumentParser(description='get_gens')
parser.add_argument('--gpu', type=str, default='0', help='which gpu id to use')
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--target_layers', help='list of layers to add the steering vector to', default = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32")
# parser.add_argument('--neg_layers', type=str, default="")
parser.add_argument('--model_path', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument('--data_path', type=str, default="data/other_domains/mathdial/test_200.json")
parser.add_argument('--sv_path', type=str, help='path to saved checkpoint')
parser.add_argument('--save_name', type=str, default="ckpt0_math")
parser.add_argument('--max_tokens', type=int, default=250)
args = parser.parse_args()

gpu_id = args.gpu.strip()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

sv = torch.load(args.sv_path)

target_layers = args.target_layers.split(" ")
target_layers = [int(i.strip()) for i in target_layers]


# negative_scaling_layers = args.neg_layers.split(" ")
# negative_scaling_layers = [int(i.strip()) for i in negative_scaling_layers]
negative_scaling_layers = []


tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map=f"cuda:0", truncation_side="left")
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map=f"cuda:0")
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Model vocab size: {model.config.vocab_size}")
print(f"ALPHA = ", args.alpha)
print(f">>>CKPT = ", args.save_name)
model.resize_token_embeddings(len(tokenizer))
max_input_tokens = model.config.max_position_embeddings

class SteeringVector(nn.Module):
    def __init__(self, hidden_size: int, dtype=torch.float16, alpha=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vector = nn.Parameter(torch.zeros(1,1, hidden_size, dtype=dtype))
        self.enabled = True
        self.alpha = alpha

    def forward(self, x):
        if self.enabled:
            return x + (self.vector.to(x.dtype)*self.alpha)
        else:
            return x
    
class SteeringLayerWrapper(nn.Module):
    def __init__(self, original_layer, config, args):
        super().__init__()
        self.original_layer = original_layer 
        self.steering_adapter = SteeringVector(config.hidden_size, alpha=args.alpha)

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.original_layer(hidden_states, *args, **kwargs)

        if isinstance(outputs, tuple):
            mod_hidden = self.steering_adapter(outputs[0])
            return (mod_hidden,) + outputs[1:]
        else:
            return self.steering_adapter(outputs)
        
def apply_steering_wrappers(model, target_layers=None):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Could not locate layers.")
    
    if target_layers is None:
        target_layers = list(range(len(layers)))
        print("Wrapping all layers")
    else:
        print(">>> Wrapping only ", target_layers)

    for i in range(len(layers)):
        if i in target_layers:
            original_layer = layers[i]

            if isinstance(original_layer, SteeringLayerWrapper):
                print(f"Layer {i} already wrapped, skipping ...")
                continue 
            wrapped_layer = SteeringLayerWrapper(original_layer, model.config, args).to(model.device)
            layers[i] = wrapped_layer
    return model



model = apply_steering_wrappers(model, target_layers=target_layers)
count = 0
for i, layer in enumerate(model.model.layers):
    # Check if this layer has been wrapped with the adapter
    if hasattr(layer, "steering_adapter"):
        if i in target_layers:
            source_tensor = sv[f'model.layers.{i}.steering_adapter.vector']
            with torch.no_grad():
                # Force copy the loaded values into this specific layer
                layer.steering_adapter.vector.copy_(source_tensor)

                if i in negative_scaling_layers: #Useful when we want to selectively subtract the vectors or skip adding them
                    layer.steering_adapter.alpha = 0#-args.alpha
                    # sign_str = "-"
                    sign_str = "0"
                else:
                    layer.steering_adapter.alpha = args.alpha
                    sign_str = "+"

            print(f"  -> Injected vector into Layer {i} with alpha: {sign_str}{args.alpha}")
            count += 1


def get_gen_selftalk(messages):
    banned_token_ids = [tokenizer.eos_token_id, 128008] # specific to llama 3.1 8B instruct - to be updated for other models
    inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
    
    outputs_before = model.generate(
            **inputs, 
            max_new_tokens=args.max_tokens,
            do_sample=False,
            temperature=0,
            use_cache=True,
        )
    actual_gen = outputs_before[0, inputs['input_ids'].shape[1]:]
    
    outputs_steered = model.generate(
            **inputs, 
            max_new_tokens=args.max_tokens,
            do_sample=False,
            temperature=0,
            use_cache=True,
            suppress_tokens=banned_token_ids  
        )
    extended_gen = outputs_steered[0, outputs_before.shape[1]:]
    gen_true = tokenizer.decode(actual_gen, skip_special_tokens=False)
    gen_extended = tokenizer.decode(extended_gen, skip_special_tokens=False)

    return gen_true, gen_extended


def prepare_sample_text(data_point, tokenizer=tokenizer):
    messages = [{"role":"system", "content":data_point['instruction']}]

    if("Counselor:" in data_point['input']):
        tokens = [t for t in re.split(r'(Client:|Counselor:)', data_point['input']) if t.strip()]
    elif("supporter:" in data_point['input']):
        tokens = [t for t in re.split(r'(seeker:|supporter:)', data_point['input']) if t.strip()]

    elif("Teacher:" in data_point['input']):
        tokens = [t for t in re.split(r'(Teacher:|Student:)', data_point['input']) if t.strip()]
    elif("Buyer:" in data_point['input']):
        tokens = [t for t in re.split(r'(Buyer:|Seller:)', data_point['input']) if t.strip()]
        # print(tokens)
    

    current_role = None
    for token in tokens:
        clean_token = token.strip()
        
        if (clean_token == "Client:") or (clean_token == "seeker:") or (clean_token == "Student:")or (clean_token == "Seller:"):
            current_role = "user"
        elif (clean_token == "Counselor:") or (clean_token == "supporter:") or (clean_token == "Teacher:")or (clean_token == "Buyer:"):
            current_role = "assistant"

        else:
            if current_role:
                messages.append({
                    "role": current_role,
                    "content": clean_token
                })
    return {'message' : messages}


train_data = load_dataset("json", data_files=args.data_path, split="train")
# train_data = train_data.select(range(200))
train_data = train_data.map(prepare_sample_text, remove_columns=train_data.column_names, batched=False)


print("Getting generations ...")
gen_trues = []
response_gen = []
for i,m in tqdm(enumerate(train_data['message']), total=len(train_data)):
    # print(m)
    gen_true, gen_extended = get_gen_selftalk(messages=m)
    gen_trues.append(gen_true)
    response_gen.append(gen_extended)

# tmp = args.saveas.split("/")
# filename = tmp[-1]
# if(len(tmp)>1):
#     tmp2 = "/".join(tmp[:-1])
#     os.makedirs(f"paper/ckpt_gens/{tmp2}", exist_ok=True)

torch.save({'gen_true':gen_trues, 'gen_extended':response_gen}, f'paper/gens_other_domain/{args.save_name}.pt')

