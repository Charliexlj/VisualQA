from datasets import Dataset
import json
import torch
import skimage.io
from transformers import AutoTokenizer
import time
from transformers import TrainingArguments, Trainer
from models import VLM, VLMConfig
import os
import random

tokenizer = AutoTokenizer.from_pretrained(...)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = 'right'

TEMPLATE = '''[INST]You are a medical multimodel tasked with question answering and interpreting chest x-ray images.

{}

[control_10]
[/INST]{}'''

INSTRUCT = [
    "Provide a detailed description of the image presented.",
    "Elaborate on the details visible in the provided image.",
    "Give an in-depth analysis of the image shown.",
    "Offer a comprehensive description of the image depicted.",
    "Examine and describe the details in the image provided.",
    "Detail the features visible in the given image.",
    "Analyze the provided image and describe its details.",
    "Provide a thorough description of the visible aspects in the image.",
    "Describe all the details you can see in the image given.",
    "Explain the details shown in the provided image in depth."
]

# Load your data from a JSON file
    
with open('openai_output/cleaned_reports_(24000-34000).json', 'r') as file:
    data_1 = json.load(file)
    
with open('openai_output/cleaned_reports_(36000-46000).json', 'r') as file:
    data_2 = json.load(file)
all_files = data_1['files'] + data_2['files']
# Create a dictionary containing your data
custom_data = {
    "text": data_1['cleaned_reports'] + data_2['cleaned_reports'], 
    "images": list(range(len(all_files)))
}

# Create a Dataset object
custom_dataset = Dataset.from_dict(custom_data)

# Define your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_template(texts):
    return[TEMPLATE.format(random.choice(INSTRUCT), t) + tokenizer.eos_token for t in texts]

# Define the tokenize function
def tokenize_function(examples):
    # Tokenize text
    text = apply_template(examples['text'])
    ids = tokenizer.batch_encode_plus(text, padding='max_length', truncation=True, return_tensors='pt', max_length=256)
    images = [[img] for img in examples['images']]
    return {'text': text,
            'input_ids': ids['input_ids'],
            'attention_mask': ids['attention_mask'],
            'labels': ids['input_ids'],
            'images': images}

# Tokenize the dataset
tokenized_dataset = custom_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True)

train_dataset = tokenized_dataset["train"].shuffle(seed=42)
eval_dataset = tokenized_dataset["test"].shuffle(seed=42)

print('Finished tokenizing dataset')
print('Train dataset length:', len(train_dataset))
print('Eval dataset length:', len(eval_dataset))
print(train_dataset[0]['text'])

vlm_config = VLMConfig()

model = VLM(vlm_config)#.to(device)
# model.disable_classification()
model.disable_classification()
model.train_proj()

def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

model.file_names = [os.path.join('/data/cl2920/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org', f) for f in all_files]

print('Model created')
import transformers
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params_all = sum(p.numel() for p in model.parameters())
print('Total trainable parameters:', total_params)
print('Total parameters:', total_params_all)

trainer = Trainer(
    model = model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args = transformers.TrainingArguments(
        per_device_train_batch_size = 1, #Batch size per GPU for training
        gradient_accumulation_steps = 8,
        num_train_epochs = 50, #Total number of training steps.(Overrides epochs)
        learning_rate = 2e-4,
        fp16 = False,
        output_dir = "/data/cl2920/Trained_Models/CLIP_align/outputs",
        optim="adamw_hf",
        warmup_steps=16,
        save_strategy='epoch',
        save_steps=100,
        save_total_limit=3,
        logging_steps = 10,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps = 500
    ),
)
trainer.train()