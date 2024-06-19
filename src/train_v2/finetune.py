from datasets import Dataset
import json
import torch
import skimage.io
from transformers import AutoTokenizer
import time
from transformers import TrainingArguments, Trainer, TrainerCallback
from models import VLM, VLMConfig
import os
import random
import numpy as np

TEMPLATE = '''[INST]You are a medical multimodel tasked with question answering and interpreting images.

Provide a detailed description of the image presented.

[control_10] [/INST]'''

seed_value = 42
random.seed(seed_value)  # Python
np.random.seed(seed_value)  # Numpy
torch.manual_seed(seed_value)  # PyTorch
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)  # PyTorch CUDA (for multi-GPU)

tokenizer = AutoTokenizer.from_pretrained(...)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = 'right'

# Load data from a JSON file
    
with open('src/train_v2/utils/final_cleaned_dialog_all_1.json', 'r') as file:
    data_1 = json.load(file)
    
with open('src/train_v2/utils/final_cleaned_dialog_all_2.json', 'r') as file:
    data_2 = json.load(file)
    
all_files = data_1['files'] + data_2['files']
all_dialog = data_1['dialog'] + data_2['dialog']

# Create a dictionary containing your data
custom_data = {
    "text": all_dialog, 
    "images": list(range(len(all_files)))
}

# Create a Dataset object
custom_dataset = Dataset.from_dict(custom_data)

# Define the tokenize function
def tokenize_function(examples):
    # Tokenize text
    ids = tokenizer.batch_encode_plus(examples['text'], padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    images = [[img] for img in examples['images']]
    return {'input_ids': ids['input_ids'],
            'attention_mask': ids['attention_mask'],
            'labels': ids['input_ids'],
            'images': images}

# Tokenize the dataset
tokenized_dataset = custom_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.02, shuffle=True)

train_dataset = tokenized_dataset["train"].shuffle(seed=42)
eval_dataset = tokenized_dataset["test"].shuffle(seed=42)

print('Finished tokenizing dataset')
print('Train dataset length:', len(train_dataset))
print('Eval dataset length:', len(eval_dataset))

model = VLM.from_pretrained('/data/cl2920/Trained_Models/CLIP_align/outputs/checkpoint-500')
# model.disable_classification()
model.finetune()

def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


model.file_names = [os.path.join('/data/cl2920/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org', f) for f in all_files]

print('Model created')

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import transformers
lora_config = LoraConfig(
    r=8,
    target_modules=[
            'down_proj',
            'o_proj',
            'k_proj',
            'q_proj',
            'gate_proj',
            'up_proj',
            'v_proj',
        ],
    modules_to_save=['lm_proj',],
    task_type = "CAUSAL_LM"
)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params_all = sum(p.numel() for p in model.parameters())
print('Setting LoRA config')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

SAMPLE_1 = train_dataset[0]
SAMPLE_2 = train_dataset[1]
SAMPLE_3 = train_dataset[2]
SAMPLE_4 = train_dataset[3]
SAMPLE_5 = train_dataset[4]

class GenerateTextCallback(TrainerCallback):
    def __init__(self, prompt, samples, n_steps=50):
        self.prompt = prompt
        self.n_steps = n_steps
        self.step_count = 0
        self.samples = samples

    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % self.n_steps == 0 and torch.cuda.current_device()==0:
            print(f"Starting text generation at step {self.step_count}")
            model = kwargs['model']
            device = model.device
            for sample in self.samples:
                input_ids = model.language_tokenizer.encode(self.prompt, return_tensors="pt").to(device)
                images = torch.tensor([sample['images']]).to(device)
                generated_text = model.generate(input_ids, images=images, max_length=256).cpu().tolist()[0]
                decoded_text =  model.language_tokenizer.decode(generated_text, skip_special_tokens=False)
                print(f"\nLabel text: {sample['text']}\n")
                print(f"\nGenerated text: {decoded_text}\n")

trainer = SFTTrainer(
    model = model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer = model.language_tokenizer,
    args = transformers.TrainingArguments(
        per_device_train_batch_size = 1, #Batch size per GPU for training
        gradient_accumulation_steps = 2,
        num_train_epochs = 50, #Total number of training steps.(Overrides epochs)
        learning_rate = 1e-4,
        fp16 = True,
        output_dir = "/data/cl2920/Trained_Models/Finetune/outputs_4",
        optim="adafactor",
        warmup_steps=16,
        save_strategy='steps',
        save_steps=2000,
        save_total_limit=50,
        logging_steps = 50,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps = 500,
    ),
    peft_config = lora_config,
    packing=True,
    formatting_func = lambda x: x,
    dataset_text_field = 'text',
    callbacks=[GenerateTextCallback(TEMPLATE, [SAMPLE_1, SAMPLE_2], n_steps=50)]
)
trainer.train()