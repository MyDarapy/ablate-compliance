import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc

#EPSILON = 1e-9  # Small value to prevent division by zero
device = "cuda" if torch.cuda.is_available() else "cpu"

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float, Int
from colorama import Fore
from dataclasses import dataclass



EPSILON = 1e-9  # Small value to prevent division by zero
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)

@dataclass
class Config:
    model_type: str = "llama3" 
    model_path: str = "./Meta-Llama-3-8B-Instruct"  
    max_new_tokens: int = 256  
    batch_size: int = 4  
    n_inst_train: int = 25  
    n_inst_test: int = 20  
    eval_n: int = 20  
    activation_layers: List[str] = ("resid_pre", "resid_mid", "resid_post") 

    model_ids = {
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama2-hf-chat": "meta-llama/Llama-2-7b-chat-hf",
        "vicuna": "lmsys/vicuna-7b-v1.3"
    }

MODEL_ID= "meta-llama/Llama-2-7b-chat-hf"


#git clone https://huggingface.co/{MODEL_ID} {MODEL_TYPE}

safety_sys = ("Follow these rules strictly. If the user asks for anything illegal, dangerous, self-harm, or hateful, "
    "refuse briefly and suggest safe alternatives. Do not provide instructions or details that enable harm.")

def get_jailbroken_suffix_instructions():
    '''dataset_1 = load_dataset("Oluwadara/adv-bench", split="train")
    dataset_2 = load_dataset("Oluwadara/adv-bench2", split="train")
    dataset = concatenate_datasets([dataset_1, dataset_2]) 
    dataset = load_dataset("Oluwadara/adv-bench2_llama", split="train") #.select(range(10))'''
    dataset = load_dataset("Oluwadara/JBB-Behaviors-harmful-nanogcg", split="train")
    train, test = dataset.train_test_split(test_size=0.5, seed=42, shuffle=True).values()
    return train, test

def get_refusal_instructions():
    '''dataset_1 = load_dataset("Oluwadara/adv-bench", split="train")
    dataset_2 = load_dataset("Oluwadara/adv-bench2", split="train")
    dataset = concatenate_datasets([dataset_1, dataset_2]) 
    dataset = load_dataset("Oluwadara/adv-bench2_llama", split="train")'''
    dataset = load_dataset("Oluwadara/JBB-Behaviors-harmful-nanogcg", split="train")
    train, test = dataset.train_test_split(test_size=0.5, seed=42, shuffle=True).values()
    return train, test


'''# Refusal template: includes a SYSTEM block that enforces refusal
def build_refusal_prompt(user_text: str, system_text: str) -> str:
    sys_block = (
        "<|start_header_id|>system<|end_header_id|>\n"
        + system_text + "<|eot_id|>"
    )
    return (
        "<|begin_of_text|>"
        + sys_block +
        "<|start_header_id|>user<|end_header_id|>\n"
        + user_text + "<|eot_id|>" +
        "<|start_header_id|>assistant<|end_header_id|>\n"
    ) 

# Compliance template: NO system safety block (or you can use a jailbreak suffix in the user text)
def build_compliance_prompt(user_text: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        + user_text + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>\n"
    ) '''

def build_refusal_prompt(user_text: str, system_text: str) -> str:
    return (
        "<s>[INST] <<SYS>>\n"
        + system_text
        + "\n<</SYS>>\n\n"
        + user_text
        + " [/INST]"
    )

def build_compliance_prompt(user_text: str) -> str:
    return (
        "<s>[INST] "
        + user_text
        + " [/INST]"
    ) 


model = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    #local_files_only=True,
    dtype=torch.bfloat16,   
    default_padding_side='left'
)

model.tokenizer.padding_side = 'left'
model.tokenizer.pad_token = model.tokenizer.eos_token

def tokenize_refusal_instructions_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str], safety_sys: str
) -> Int[Tensor, 'batch_size seq_len']:
    prompts = [build_refusal_prompt(instruction, safety_sys) for instruction in instructions]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

def tokenize_compliance_instructions_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str]
) -> Int[Tensor, 'batch_size seq_len']:
    prompts = [build_compliance_prompt(instruction) for instruction in instructions]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

def tokenize_refusal_instructions_fn(*, instructions: List[str]) -> torch.Tensor:
    return tokenize_refusal_instructions_chat(
        tokenizer=model.tokenizer,
        instructions=instructions, safety_sys=safety_sys,
    )

def tokenize_compliance_instructions_fn(*, instructions: List[str]) -> torch.Tensor:
    return tokenize_compliance_instructions_chat(
        tokenizer=model.tokenizer,
        instructions=instructions,
    )

def tokenize_pair(
    refusal_instructions: List[str],
    jailbroken_instructions: List[str]
) -> Int[Tensor, 'batch seq']:
    # BYPASS WRAPPERS: call the chat funcs directly with explicit args
    r = tokenize_refusal_instructions_chat(
        tokenizer=model.tokenizer,
        instructions=refusal_instructions,
        safety_sys=safety_sys,
    )
    j = tokenize_compliance_instructions_chat(
        tokenizer=model.tokenizer,
        instructions=jailbroken_instructions,
    )

    # pad to common length then concat
    max_len = max(r.shape[1], j.shape[1])
    def pad_to(x, L):
        if x.shape[1] == L: return x
        pad = torch.full((x.shape[0], L - x.shape[1]), model.tokenizer.pad_token_id, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)
    r = pad_to(r, max_len)
    j = pad_to(j, max_len)
    return torch.cat([r, j], dim=0)

def _generate_with_hooks(
    model,
    toks: torch.Tensor,  # [B, S]
    max_tokens_generated: int = 64,
    fwd_hooks=(),
) -> List[str]:
    toks = toks.to(model.W_E.device if hasattr(model, "W_E") else toks.device)
    pad_id = model.tokenizer.pad_token_id
    assert pad_id is not None, "pad_token_id must be set."

    B, S = toks.shape
    all_toks = torch.full((B, S + max_tokens_generated), pad_id, dtype=torch.long, device=toks.device)
    all_toks[:, :S] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :S + i])  # only feed actual prefix
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        all_toks[:, S + i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, S:], skip_special_tokens=True)


def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 100,
    batch_size: int = 4,
) -> List[str]:
    generations = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    return generations

try:
    del refusal_logits
except Exception:
    pass
try:
    del jailbroken_logits
except Exception:
    pass
gc.collect(); torch.cuda.empty_cache()

"""Finding potential "jailbreaking/compliance directions" (batched)"""

refusal = {}
jailbreak = {}

N_INST_TRAIN = 25

refusal_instruction_train, refusal_instruction_test = get_refusal_instructions()
jailbroken_instruction_train, jailbroken_instruction_test = get_jailbroken_suffix_instructions()

refusal_instructions_train = [x['message'] for x in refusal_instruction_train]
refusal_instructions_test = [x['message'] for x in refusal_instruction_test]

jailbroken_instructions_train = [x['message']+ " " + x['result'] for x in jailbroken_instruction_train]
jailbroken_instructions_test = [x['message']+ " " + x['result'] for x in jailbroken_instruction_test]

# Tokenize both halves together and then split cleanly
toks = tokenize_pair(
    refusal_instructions_train[:N_INST_TRAIN],
    jailbroken_instructions_train[:N_INST_TRAIN]
).to(device)

refusal_toks,jailbroken_toks = toks.split(N_INST_TRAIN)

batch_size = 4

for i in tqdm(range(0, N_INST_TRAIN // batch_size + (N_INST_TRAIN % batch_size > 0))):
    id = i*batch_size
    e = min(N_INST_TRAIN,id+batch_size)

    # run the models on refusal and compliance/jailbreak prompts, cache their activations separately.
    refusal_logits, refusal_cache = model.run_with_cache(refusal_toks[id:e], names_filter=lambda hook_name: 'resid' in hook_name, reset_hooks_end=True)
    jailbroken_logits, jailbroken_cache = model.run_with_cache(jailbroken_toks[id:e], names_filter=lambda hook_name: 'resid' in hook_name, reset_hooks_end=True)

    for key in refusal_cache:
        if key not in refusal:
            refusal[key] = [refusal_cache[key]]
            jailbreak[key] = [jailbroken_cache[key]]
        else:
            refusal[key].append(refusal_cache[key])
            jailbreak[key].append(jailbroken_cache[key])

    # force Python & PyTorch to clear GPU and CPU RAM where possible
    del refusal_logits, jailbroken_logits, refusal_cache, jailbroken_cache
    gc.collect()
    torch.cuda.empty_cache()

refusal = {k:torch.cat(v) for k,v in refusal.items()}
jailbroken = {k:torch.cat(v) for k,v in jailbreak.items()}
#print(f"refusal_dict keys: {list(refusal.values())}")
#print(f"jailbreak_dict keys: {list(jailbreak.values())}")


# compute difference of means between compliance and refused activations at intermediate layers

def get_act_idx(cache_dict, act_name, layer):
    key = (act_name, layer,)
    return cache_dict[utils.get_act_name(*key)]

#activation_layers = ['resid_pre', 'resid_mid', 'resid_post']

activation_layers = ['resid_post'] 

activation_jailbreaks = {k:[] for k in activation_layers}

for layer_num in range(1,model.cfg.n_layers):
    pos = -1 # last token in the prompt (before generation starts)

    for layer in activation_layers:
        refusal_mean_act = get_act_idx(refusal, layer, layer_num)[:, pos, :].mean(dim=0)
        jailbroken_mean_act = get_act_idx(jailbroken, layer, layer_num)[:, pos, :].mean(dim=0)

        jailbreak_direction = jailbroken_mean_act - refusal_mean_act #the jailbreak axis
        jailbreak_direction = jailbreak_direction / jailbreak_direction.norm()
        activation_jailbreaks[layer].append(jailbreak_direction)

print(activation_jailbreaks)
# save to file so you don't have to re-build later
torch.save(activation_jailbreaks, 'jailbreak_dirs.pth')
jailbreak_dirs = activation_jailbreaks

# Ablate (remove) "jailbreak direction" via inference-time intervention such that give the adversial suffix in the prompt the model still refuses
# Get all calculated potential jailbreak dirs, sort them in Descending order (reverse) based on their mean()


activation_layers = ['resid_pre', 'resid_mid', 'resid_post'] # you can use a subset of these if you don't think certain activations are promising

activation_layers = ['resid_post']
activation_scored = sorted([activation_jailbreaks[layer][l-1] for l in range(1,model.cfg.n_layers) for layer in activation_layers], key = lambda x: abs(x.mean()), reverse=True)


#### Model ablation testing/brute-forcing the best compliance dir
##### Inference-time intervention hook:


def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

print(f"JAILBREAKS BASELINE WITH A BASE MODEL NO INTERVENTIONS YET")
N_INST_TEST = 20
baseline_generations = get_generations(
    model,
    jailbroken_instructions_test[:N_INST_TEST],
    tokenize_compliance_instructions_fn,
    fwd_hooks=[]
)
# and when you need jailbreak prompts:
# get_generations(model, jailbroken_instructions_test[:N_INST_TEST], tokenize_compliance_instructions_fn, fwd_hooks=fwd_hooks)

for gen in baseline_generations:
    print(gen)


#Evaluating layers defined earlier (needs human evaluation to determine best layer for jailbreak inhibition)

if "N_INST_TEST" not in locals() or not N_INST_TEST:
    N_INST_TEST = 10
EVAL_N = 20 
evals = []

for jailbreak_dir in tqdm(activation_scored[:EVAL_N]):
    intervention_layers = list(range(model.cfg.n_layers)) # all layers

    hook_fn = functools.partial(direction_ablation_hook,direction=jailbreak_dir)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

    intervention_generations = get_generations(
        model,
        jailbroken_instructions_test[:N_INST_TEST],
        tokenize_compliance_instructions_fn,
        fwd_hooks=fwd_hooks
    )
    evals.append(intervention_generations)

    print(intervention_generations) 

#### Present evals to clever pre-trained non-refusing human

for instruction in range(N_INST_TEST):
    if 'baseline_generations' in locals() and baseline_generations and len(baseline_generations) > instruction:
        print(f"INSTRUCTION {instruction}: {repr(jailbroken_instructions_test[instruction])}")
        print(Fore.GREEN + f"BASELINE COMPLETION:")
        print(textwrap.fill(repr(baseline_generations[instruction]), width=100, initial_indent='\t', subsequent_indent='\t'))
    for layer_candidate in range(EVAL_N):
        if len(evals) > layer_candidate and len(evals[layer_candidate]) > instruction:
            print(Fore.RED + f"LAYER CANDIDATE #{layer_candidate} INTERVENTION COMPLETION:")
            print(textwrap.fill(repr(evals[layer_candidate][instruction]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)

"""## Orthogonalize weights w.r.t. "jailbreak direction" """
layer_candidate = 9
jailbreak_dir = activation_scored[layer_candidate]

### Write ortho'd weights into model

def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[Tensor, '... d_model']:
    proj = einops.einsum(matrix, vec.view(-1, 1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj

if jailbreak_dir.device != model.W_E.device:
    jailbreak_dir = jailbreak_dir.to(model.W_E.device)
model.W_E.data = get_orthogonalized_matrix(model.W_E, jailbreak_dir)

for block in tqdm(model.blocks):
    if jailbreak_dir.device != block.attn.W_O.device:
        jailbreak_dir = jailbreak_dir.to(block.attn.W_O.device)
    block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O, jailbreak_dir)
    block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out, jailbreak_dir)

# save your jailbreak_dir of choice separately to a file
torch.save(jailbreak_dir,"ablation.pth")

### Verify model weights are adjusted to match ablation (skippable)

orthogonalized_generations = get_generations(
    model,
    jailbroken_instructions_test[:N_INST_TEST],
    tokenize_compliance_instructions_fn,
    fwd_hooks=[]
)

for i in range(N_INST_TEST):
    if 'baseline_generations' in locals() and baseline_generations and len(baseline_generations) > i:
        print(f"INSTRUCTION {i}: {repr(jailbroken_instructions_test[i])}")
        print(Fore.GREEN + f"BASELINE COMPLETION:")
        print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RED + f"INTERVENTION COMPLETION:")
    print(textwrap.fill(repr(evals[layer_candidate][i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.MAGENTA + f"ORTHOGONALIZED COMPLETION:")
    print(textwrap.fill(repr(orthogonalized_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)

torch.save(model, "pytorch_model.bin") 
cfg = model.cfg

state_dict = model.state_dict()

hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=torch.bfloat16) # load the original model as a regular unhooked Transformer -- don't need to load it into GPU as it's just for saving
lm_model = hf_model.model

#### Llama-3 conversion 

lm_model.embed_tokens.weight = torch.nn.Parameter(state_dict["embed.W_E"].cpu())

#for l in range(cfg.n_layers):
#    lm_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter(einops.rearrange(state_dict[f"blocks.{l}.attn.W_O"], "n h m->m (n h)", n=cfg.n_heads).contiguous())
#    lm_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(torch.transpose(state_dict[f"blocks.{l}.mlp.W_out"],0,1).contiguous())
    
#### Save converted model

#hf_model.save_pretrained("path/to/my/")
#hf_model.push_to_hub("Oluwadara/llama2-chat-hardened") """

