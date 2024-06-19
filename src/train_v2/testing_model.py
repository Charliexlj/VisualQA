import json
import torch
import skimage.io
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import time
from transformers import TrainingArguments, Trainer
from models import VLM, VLMConfig
import os
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained(...)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = 'right'

TEMPLATE = '''[INST]You are a medical multimodel tasked with question answering and interpreting images.

Provide a detailed description of the image presented.

[control_10] [/INST]'''

No_Image_TEMPLATE = '''[INST] {} [/INST]'''

# Load your data from a JSON file
with open('src/train_v2/utils/cleaned_reports.json', 'r') as file:
    data = json.load(file)

# Create a dictionary containing your data
custom_data = {
    "text": data['cleaned_reports'], 
    "images": list(range(len(data['files'])))
}

# Create a Dataset object
custom_dataset = Dataset.from_dict(custom_data)

# Define your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_template(texts):
    return[TEMPLATE.format(t) for t in texts]

# Define the tokenize function
def tokenize_function(examples):
    # Tokenize text
    ids = tokenizer.batch_encode_plus(apply_template(examples['text']), padding='max_length', truncation=True, return_tensors='pt', max_length=256)
    images = [[img] for img in examples['images']]
    return {'input_ids': ids['input_ids'],
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

model = VLM.from_pretrained('/data/cl2920/Trained_Models/CLIP_align/outputs/checkpoint-500/')
model = PeftModel.from_pretrained(model, '/data/cl2920/Trained_Models/Finetune/outputs_5/checkpoint-4000')
model = model.merge_and_unload()
# model.disable_classification()
model = model.to(device)
model.eval()
model.file_names = [os.path.join('/data/cl2920/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org', f) for f in data['files']]


while input('Do you want to continue? (y/n) ') == 'y':
    image = None
    chat_dict=None
    # if input('Do you want to include an image? (y/n) ') == 'y':
    idx = torch.randint(0, len(data['files']), (1,)).item()
    print('File name:', data['files'][idx])
    image = skimage.io.imread(os.path.join('/data/cl2920/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org', data['files'][idx]))
    if image.ndim == 2:
        image = image[None]
    image = torch.tensor(image).unsqueeze(0).to(device)
    text = data['cleaned_reports'][idx]
    print('\nImage Original Report:', text, '\n')
    chat_dict = model.conversation("Provide a detailed description of the image presented.", image, chat_dict)
    print('Generated text:', chat_dict['output'],   '\n')
    chat_dict = model.conversation(input('Second Question: '), None, chat_dict)
    print('Generated text:', chat_dict['output'],   '\n')
    chat_dict = model.conversation(input('Third Question: '), None, chat_dict)
    print('Generated text:', chat_dict['output'],   '\n')