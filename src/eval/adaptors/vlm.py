import json
import torch
import skimage.io
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import time
from transformers import TrainingArguments, Trainer
import os
from datasets import Dataset
import sys
from tqdm import tqdm

import warnings

# Suppress specific warnings from transformers
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set.*")

# Get the directory of the current script
current_script_directory = os.path.dirname(os.path.abspath(__file__))
print("Current Script Directory:", current_script_directory)

# Construct the path to the 'models' directory
models_path = '/homes/cl2920/Projects/CheXmed-VisualQA/src/train_v2/models'
print("Models Path:", models_path)

# Check if the path actually exists
if os.path.exists(models_path):
    print("Confirmed: The models path exists.")
else:
    print("Error: The models path does not exist.")

# Add the models path to sys.path
sys.path.append(models_path)
print("sys.path:", sys.path)

from models.HF_Model import VLM, VLMConfig

PROMPT = '''[INST]You are a medical multimodel tasked with question answering and interpreting images.

Provide a detailed description of the image presented.

[control_10] [/INST]'''

def VLM_Findings(img_paths, device='cuda'):
    all_outputs = []
    model = VLM.from_pretrained('/data/cl2920/Trained_Models/CLIP_align/outputs/checkpoint-500/')
    model = PeftModel.from_pretrained(model, '/data/cl2920/Trained_Models/Finetune/outputs_5/checkpoint-5000')
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    model.file_names = img_paths
    
    for i, path in tqdm(enumerate(img_paths)):
        img = skimage.io.imread(path)
        img = torch.tensor(img).to(device).float()
        if len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        else:
            if img.shape[0] in [1, 3]:
                img = torch.mean(img, dim=0).unsqueeze(0).unsqueeze(0)
            else:
                img = torch.mean(img, dim=2).unsqueeze(0).unsqueeze(0)
    
        input_ids = model.language_tokenizer.encode(PROMPT, return_tensors='pt').to(device)
        mask = torch.ones_like(input_ids).to(device)
        outputs = model.generate(input_ids, images=img, attention_mask=mask, max_length=256, eos_token_id=model.language_tokenizer.eos_token_id, pad_token_id=model.language_tokenizer.eos_token_id)
        outputs = outputs[0][len(input_ids[0]):]
        outputs = model.language_tokenizer.decode(outputs, skip_special_tokens=True)
        all_outputs.append(outputs)
    
    return all_outputs
        
    