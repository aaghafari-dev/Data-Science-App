from __future__ import annotations
import os
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import config

class LocalLLMLoader:
    def __init__(self): self.loaded_model=None; self.loaded_model_name=None
    def load_model(self, model_name):
        if model_name not in config.LOCAL_LLM_MODELS: return None
        cache_path=os.path.join(config.HF_CACHE_DIR,config.LOCAL_LLM_MODELS[model_name]); snapshots=os.path.join(cache_path,"snapshots")
        if os.path.exists(snapshots):
            folders=[x for x in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots,x))]; model_path=os.path.join(snapshots,folders[0]) if folders else cache_path
        else: model_path=cache_path
        try:
            kwargs={}
            use_cuda=False
            if torch.cuda.is_available():
                try: use_cuda = torch.cuda.get_device_properties(0).total_memory >= 4 * 1024**3
                except Exception: use_cuda = False
            if use_cuda:
                # Small GPUs such as MX250 are intentionally kept on CPU for local LLM reporting.
                kwargs={"device_map":"auto","dtype":torch.float16}
            else: kwargs={"device_map":"cpu","dtype":torch.float32}
            try: model=AutoModelForCausalLM.from_pretrained(model_path,**kwargs)
            except Exception:
                try: model=AutoModelForSeq2SeqLM.from_pretrained(model_path,**kwargs)
                except Exception:
                    if torch.cuda.is_available():
                        try: torch.cuda.empty_cache()
                        except Exception: pass
                    try: model=AutoModelForCausalLM.from_pretrained(model_path,device_map="cpu",dtype=torch.float32)
                    except Exception: model=AutoModelForSeq2SeqLM.from_pretrained(model_path,device_map="cpu",dtype=torch.float32)
            tokenizer=AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
            pipe_kind="text2text-generation" if "flan" in model_name.lower() else "text-generation"
            pipe=pipeline(pipe_kind,model=model,tokenizer=tokenizer,max_new_tokens=256,temperature=0.2,do_sample=False,pad_token_id=tokenizer.pad_token_id)
            self.loaded_model=HuggingFacePipeline(pipeline=pipe); self.loaded_model_name=model_name; return self.loaded_model
        except Exception as e:
            print(f"Error loading local model {model_name}: {e}"); return None
    def get_llm(self, model_name):
        return self.loaded_model if self.loaded_model_name==model_name else self.load_model(model_name)
