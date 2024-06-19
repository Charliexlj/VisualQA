import json
import torch
from transformers import BertTokenizer

# Load the JSON file
with open('./output/llama_8B_instruct/labels.json', 'r') as file:
    data = json.load(file)

concise_reports = data['concise_reports']

tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

tokenized_concise_reoprts = []
for text in concise_reports:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.tolist() for k, v in inputs.items()}
    tokenized_concise_reoprts.append(inputs)
    
output_data = {"files": data['files'], 
               "tokenized_concise_reports": tokenized_concise_reoprts}

with open('./src/train/utils/clip_pretrain.json', 'w') as file:
    json.dump(output_data, file, indent=4)


    

