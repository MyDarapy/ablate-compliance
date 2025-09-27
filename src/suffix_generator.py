import nanogcg
import torch
from datasets import load_dataset, Dataset
from nanogcg import GCGConfig
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.cache_utils import DynamicCache

def _wrap_forward_cache(model):
    orig_forward = model.forward
    def forward(*args, **kwargs):
        pkv = kwargs.get("past_key_values", None)
        # nanogcg passes a list/tuple (legacy). Convert once at the boundary.
        if isinstance(pkv, (list, tuple)):
            kwargs["past_key_values"] = DynamicCache.from_legacy_cache(pkv)
        return orig_forward(*args, **kwargs)
    model.forward = forward


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  
#model_id = "meta-llama/Llama-2-7b-chat-hf"  
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
_wrap_forward_cache(model)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0

)

dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful").select(range(20))
output = []
with open("llama3_8B.jsonl", "a") as f:   
    for i in tqdm(range(len(dataset))):
        message = dataset[i]["Goal"]
        target = dataset[i]["Target"]

        config = GCGConfig(
            num_steps=200,
            search_width=64,
            topk=64,
            seed=42,
            verbosity="WARNING"
        )

        result = nanogcg.run(model, tokenizer, message, target, config)
    
        q = message + " " + result.best_string
        out = pipe(q, max_new_tokens=100, do_sample=False)[0]["generated_text"]
        ans = out
        print("="*60)
        print("Q:", q)
        print("Model:", ans) 

        entry = {
            "message": message,
            "target": target, 
            "result": result.best_string,
            "model_answer": ans
        }
        output.append(entry)
        f.write(json.dumps(entry) + "\n")
        f.flush()

out_ds = Dataset.from_list(output)
#out_ds.save_to_disk("suffix_breaker_harmbench ")
out_ds.push_to_hub("Oluwadara/llama3")

