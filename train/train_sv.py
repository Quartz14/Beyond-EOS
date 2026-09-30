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
    AutoTokenizer,
    TrainingArguments,
    logging,
    set_seed,
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import re
import time

start_time = time.perf_counter()

parser = argparse.ArgumentParser(description='vector_sft')
parser.add_argument('--gpu', type=str, default='0', help='which gpu id to use')
parser.add_argument("--inst", type=str, default='none', help="choose system prompt according to dataset, supported options: ['default', 'cbt_short', 'cbt_short', 'es_long', 'escot_long']")
parser.add_argument("--output_dir", type=str)
parser.add_argument('--base_model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument('--dataset_name', type=str, default="data/cactus", help='path to dataset folder')
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_steps', type=int, default=1000)
parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
parser.add_argument('--run_name', type=str, default='sample_run', help='name to save the checkpoints under')
parser.add_argument('--output_col', type=str, default='output')

args = parser.parse_args()

output_dir = f'./checkpoints/{args.output_dir}/{args.run_name}'

max_steps = int(args.max_steps)
print(">>> Max steps: ", max_steps)
gpu_id = args.gpu.strip()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

set_seed(2025)
os.makedirs(output_dir, exist_ok=True)
logging.set_verbosity_error()

class SteeringVector(nn.Module):
    def __init__(self, hidden_size: int, dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.vector = nn.Parameter(torch.zeros(1,1, hidden_size, dtype=dtype))
        self.enabled = True

    def forward(self, x):
        if self.enabled:
            return x + self.vector.to(x.dtype)
        else:
            return x

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )      

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    max_token_length = 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
            if len(tokenizer(text).tokens()) > max_token_length:
                max_token_length = len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
            if len(tokenizer.tokenize(text)) > max_token_length:
                max_token_length = len(tokenizer.tokenize(text))

    print(f"max token length: {max_token_length}")
    return total_characters / total_tokens


class SteeringLayerWrapper(nn.Module):
    def __init__(self, original_layer, config):
        super().__init__()
        self.original_layer = original_layer 
        self.steering_adapter = SteeringVector(config.hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.original_layer(hidden_states, *args, **kwargs)

        if isinstance(outputs, tuple):
            mod_hidden = self.steering_adapter(outputs[0])
            return (mod_hidden,) + outputs[1:]
        else:
            return self.steering_adapter(outputs)
        

def apply_steering_wrappers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Could not locate layers.")
    print(f"Wrapping {len(layers)} layers...")
    for i in range(len(layers)):
        original_layer = layers[i]
        wrapped_layer = SteeringLayerWrapper(original_layer, model.config).to(model.device)
        layers[i] = wrapped_layer
    return model

tokenizer =  AutoTokenizer.from_pretrained(args.base_model, truncation_side="left")
tokenizer.pad_token = tokenizer.eos_token



def prepare_sample_text(data_point, tokenizer=tokenizer, args=args):

    if(args.inst =='default'):
        messages = [{"role":"system", "content":data_point['instruction']}]
    elif(args.inst == 'cbt_short'):
        messages = [{"role":"system", "content":'Generate the response as the counselor.'}]
    elif(args.inst == 'es_long'):
        messages = [{"role":"system", "content":'You are playing the role of a supporter in a emotion support conversation session. Your task is to generate the next supporter utterance in the dialogue. The goal is to create a natural and engaging response that builds on the previous conversation.'}]
    elif(args.inst == 'escot_long'):
        messages = [{"role":"system", "content":'You are playing the role of a supporter in a emotion support conversation session. Your task is to generate the response as the supporter using the pipeline of Emotion, Emotion Stimulus, Individual Appraisal, Strategy Reason, Response. The goal is to create a natural and engaging response that builds on the previous conversation.'}]
    elif(args.inst == 'none'):
        messages = []

    if("Counselor:" in data_point['input']):
        tokens = [t for t in re.split(r'(Client:|Counselor:)', data_point['input']) if t.strip()]
    elif("supporter:" in data_point['input']):
        tokens = [t for t in re.split(r'(seeker:|supporter:)', data_point['input']) if t.strip()]
    elif("Teacher:" in data_point['input']):
        tokens = [t for t in re.split(r'(Teacher:|Student:)', data_point['input']) if t.strip()]
    elif("Buyer:" in data_point['input']):
        tokens = [t for t in re.split(r'(Buyer:|Seller:)', data_point['input']) if t.strip()]

    current_role = None
    for token in tokens:
        clean_token = token.strip()
        
        if (clean_token == "Client:") or (clean_token == "seeker:") or (clean_token == "Student:") or (clean_token == "Seller:"):
            current_role = "user"
        elif (clean_token == "Counselor:") or (clean_token == "supporter:") or (clean_token == "Teacher:") or (clean_token == "Buyer:"):
            current_role = "assistant"

        else:
            if current_role:
                messages.append({
                    "role": current_role,
                    "content": clean_token
                })
    messages.append({'role':'assistant', 'content':data_point[args.output_col]})
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=False, # Set False for training (we already added the assistant response)
    )

    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        max_length=4096, # Set your desired max sequence length
        padding=False,   # Important: Do not pad here
        add_special_tokens=False, # apply_chat_template usually adds them already
        
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }


response_template_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
instruction_template_str = "<|start_header_id|>user<|end_header_id|>\n\n"

class MultiTurnResponseCollator:
    def __init__(self, tokenizer, response_template, instruction_template, mlm=False):
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # Helper to encode template strings to IDs
        def to_ids(template):
            if isinstance(template, str):
                return tokenizer.encode(template, add_special_tokens=False)
            return template

        self.response_token_ids = to_ids(response_template)
        self.instruction_token_ids = to_ids(instruction_template)

    def __call__(self, examples):
        # 1. Pad inputs using the base collator
        batch = self.base_collator(examples)
        input_ids = batch["input_ids"]
        
        # 2. Initialize labels to -100 (MASK EVERYTHING by default)
        labels = torch.full_like(input_ids, -100)
        
        # 3. Iterate over each sequence in the batch
        for i in range(len(input_ids)):
            seq = input_ids[i]
            
            # Find all indices where the Assistant (Response) starts
            response_starts = self._find_all(seq, self.response_token_ids)
            
            # Find all indices where the User (Instruction) starts
            instruction_starts = self._find_all(seq, self.instruction_token_ids)
            
            # --- The Logic: Unmask segments belonging to Assistant ---
            for start_idx in response_starts:
                # The actual answer starts after the header
                answer_start = start_idx + len(self.response_token_ids)
                
                # Find the NEXT instruction start (or end of sequence)
                # This marks where the assistant's turn ends.
                next_instruction = len(seq) # Default to end of sequence
                for inst_idx in instruction_starts:
                    if inst_idx > answer_start:
                        next_instruction = inst_idx
                        break
                
                # Unmask this range (Set labels = input_ids)
                # We stop slightly before the next instruction to avoid learning the next header
                labels[i, answer_start:next_instruction] = seq[answer_start:next_instruction]

        batch["labels"] = labels
        return batch

    def _find_all(self, haystack, needle):
        """Helper to find ALL start indices of a subsequence"""
        indices = []
        n_len = len(needle)
        h_len = len(haystack)
        if n_len > h_len: return indices
        
        needle_tensor = torch.tensor(needle, device=haystack.device)
        
        # Simple sliding window (for loop is fast enough for batch sizes < 64)
        for i in range(h_len - n_len + 1):
            if torch.equal(haystack[i : i + n_len], needle_tensor):
                indices.append(i)
        return indices
    
def create_datasets(tokenizer, args):
    train_json_path = os.path.join(args.dataset_name, "train.json")
    train_data = load_dataset("json", data_files=train_json_path, split="train")
    train_data = train_data.shuffle(seed=2025)

    train_data = train_data.map(prepare_sample_text, remove_columns=train_data.column_names, batched=False)


    val_json_path = os.path.join(args.dataset_name, "val.json")
    valid_data = load_dataset("json", data_files=val_json_path, split="train")
    valid_data = valid_data.shuffle(seed=2025)
    valid_data = valid_data.map(prepare_sample_text, remove_columns=valid_data.column_names, batched=False)

    return train_data, valid_data

def get_gen(model, tokenizer, messages):
    banned_token_ids = [tokenizer.eos_token_id]
    
    # Ensure model is in eval mode for generation
    was_training = model.training
    model.eval()
    
    inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
        
    outputs_steered = model.generate(
            **inputs, 
            max_new_tokens=500,
            do_sample=False,
            temperature=0,
            use_cache=True,
            suppress_tokens=banned_token_ids
        )
    gen_steered = tokenizer.decode(outputs_steered[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Switch back to training mode if it was training
    if was_training:
        model.train()
        
    return gen_steered

class SaveSteeringCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Trigger every 50 steps
        if state.global_step % 50 == 0 and state.global_step > 0:
            
            # Extract only the steering parameters
            steering_state_dict = {}
            # Unwrap model in case of DDP/DataParallel to access named_parameters correctly
            _model = model.module if hasattr(model, "module") else model
            
            for name, param in _model.named_parameters():
                if "steering_adapter" in name:
                    steering_state_dict[name] = param.cpu()
            
            step_output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(step_output_dir, exist_ok=True)
            
            # Save
            save_path = os.path.join(step_output_dir, "sv.pt")
            torch.save(steering_state_dict, save_path)
            print(f"\nSteering vectors saved to {save_path}")

class SampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, messages):
        self.tokenizer = tokenizer
        self.messages = messages
        
    # This triggers after every evaluation loop
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            model = kwargs.get('model')
            
        print(f"\n\n=== GENERATION AT STEP {state.global_step} ===")
        try:
            generated_text = get_gen(model, self.tokenizer, self.messages)
            print(f"Output:\n{generated_text}")
        except Exception as e:
            print(f"Generation failed: {e}")
        print("==========================================\n")

test_messages = [{"role":"system", "content":'Act as a value-driven, low-anchoring buyer for the Item: bike. Item description: Vintage Peugeot Road Bike Great vintage 1970\'s Peugeot 10-speed made in France. Bike is in good condition, nice decals, good tires & tubes. 22" frame on 27" wheels old, Puegeot, Peugot, french, classic, old school, retro, rare, Seller price: $160, Your target price: $147. Use a reluctance-based, incremental concession strategy to negotiate with the seller to purchase the item as close to your target price as possible.'}, {'role':'Seller', "content":"I would like 160"}]
if(args.inst == 'es_long'):
    test_messages = [{"role":"system", "content":  "You are playing the role of a supporter in a emotion support conversation session. Your task is to generate the next supporter utterance in the dialogue. The goal is to create a natural and engaging response that builds on the previous conversation."}, {"role":"user", "content": "I went to the capital of Italy and met a long lost friend unexpectedly."}]
elif(args.inst == 'escot_long'):
    test_messages = [{"role":"system", "content":  "You are playing the role of a supporter in a emotion support conversation session. Your task is to generate the response as the supporter using the pipeline of Emotion, Emotion Stimulus, Individual Appraisal, Strategy Reason, Response. The goal is to create a natural and engaging response that builds on the previous conversation."}, {"role":"user", "content": "I went to the capital of Italy and met a long lost friend unexpectedly."}]    
elif(args.inst == 'none'):
    test_messages = [{"role":"user", "content": "I went to the capital of Italy and met a long lost friend unexpectedly."}]    


model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=f"cuda:0",
        torch_dtype=torch.float16,
    )
model = apply_steering_wrappers(model)

max_input_tokens = model.config.max_position_embeddings

for name, param in model.named_parameters():
    if "steering_adapter" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

print_trainable_parameters(model)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {trainable_params}")
print(f"Expected: {len(model.model.layers)* model.config.hidden_size}")


my_collator = MultiTurnResponseCollator(
    tokenizer=tokenizer,
    response_template=response_template_str,
    instruction_template=instruction_template_str)


train_dataset, eval_dataset = create_datasets(tokenizer, args)
print(train_dataset[0])

training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        eval_strategy="steps",
        max_steps=max_steps,
        eval_steps=50,
        save_steps=50,
        logging_steps=50,
        save_total_limit=60,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=100,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        fp16=True,
        # bf16=True,
        weight_decay=0.05,
        warmup_ratio=0.,
        run_name=args.run_name,
        report_to="wandb",
    )

sample_callback = SampleGenerationCallback(tokenizer, test_messages)
steering_save_callback = SaveSteeringCallback()

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, 
                  data_collator=my_collator, callbacks=[sample_callback, steering_save_callback])

print("Training...")
trainer.train()#resume_from_checkpoint=args.resume_from_checkpoint)

steering_state_dict = {}
for name, param in model.named_parameters():
    if "steering_adapter" in name:
        steering_state_dict[name] = param.cpu()

torch.save(steering_state_dict, f"{output_dir}/sv_{max_steps}.pt") #
print(f"\n1k Steering vectors saved to {output_dir}/sv_{max_steps}.pt")


end_time = time.perf_counter()
elapsed_time = (end_time - start_time)/60
print(f"Elapsed time: {elapsed_time:.4f} minutes")
