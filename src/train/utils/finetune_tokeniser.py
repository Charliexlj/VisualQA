import json
import torch
from transformers import BertTokenizer
import random
from tqdm import tqdm
CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']
CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}
TEMPLATE = {'Positive': ["evidence of {}",
                         "positive for {}",
                         "findings consistent with {}",
                         "{} are present",
                         "there are {}"],
            'Negative': ["no evidence of {}",
                         "negative for {}",
                         "nindings inconsistent with {}",
                         "{} are absent",
                         "nthere are no {}"],
            'Uncertain': ["possible {}",
                          "suspicious for {}",
                          "questionable for {}",]}

# Load the JSON file
with open('./output/llama_8B_instruct/new_labels.json', 'r') as file:
    data = json.load(file)
    
y_labels = data['raw_label']
    
    
# Version 1, single image with multiple captions, will experience mode collapse
concise_reports = data['concise_reports']                

tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

tokenized_concise_reports = []
for text in tqdm(concise_reports):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.tolist() for k, v in inputs.items()}
    tokenized_concise_reports.append(inputs)
    
output_data = {"files": data['files'], 
               "tokenized_concise_reports": tokenized_concise_reports,
               "labels": data['raw_label'],}

with open('./src/train/utils/clip_finetune.json', 'w') as file:
    json.dump(output_data, file, indent=4)
