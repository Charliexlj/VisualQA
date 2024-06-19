import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image   
import os
import yaml
import sys
from tqdm import tqdm
import math
import json

with open('./src/eval/eval_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
sys.path.append(config["model_path"])

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_tokenizer(tokenizer_path, max_img_size = 100, image_num = 32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to 
    '''
    if isinstance(tokenizer_path,str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        special_token = {"additional_special_tokens": ["<image>","</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image"+str(i*image_num+j)+">")
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(
                special_token
            )
            ## make sure the bos eos pad tokens are correct for LLaMA-like models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2    
    
    return  text_tokenizer,image_padding_tokens    

def combine_and_preprocess(question,image_list,image_padding_tokens):
    
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    images  = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']
        
        image = Image.open(img_path).convert('RGB')   
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1) # c,w,h,d
        
        ## pre-process the img first
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size = (target_H,target_W,target_D)))
        
        ## add img placeholder to text
        new_qestions[position] = "<image>"+ image_padding_tokens[padding_index] +"</image>" + new_qestions[position]
        padding_index +=1
    
    vision_x = torch.cat(images,dim = 1).unsqueeze(0) #cat tensors and expand the batch_size dim
    text = ''.join(new_qestions) 
    return text, vision_x, 
    
def print_memory_usage(device_id):
    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved(device_id) / (1024**3)  # GB
    print(f"Device {device_id}: Allocated memory = {allocated:.2f} GB, Reserved memory = {reserved:.2f} GB")

    
def RadFM_Findings(img_paths, model_path, question="Provide a detailed description of the given chest X-ray image, resembling a comprehensive radiology report."):
    FOLDER_PATH = os.path.join(model_path, 'Quick_demo')
    
    print("Setup tokenizer")
    tokenizer_path = os.path.join(FOLDER_PATH, 'Language_files')
    text_tokenizer,image_padding_tokens = get_tokenizer(tokenizer_path)
    print("Finish loading tokenizer")
    
    print("Setup Model")
    num_layers = read_json(os.path.join(os.path.join(FOLDER_PATH, 'Language_files'), "config.json"))["num_hidden_layers"]
    device_ids = ['cuda:0', 'cuda:1']
    print("Device IDs: ",device_ids)
    device_map = {
        "model.embed_tokens": device_ids[0],
        "model.norm.weight": device_ids[-1],
        "lm_head": device_ids[-1],
    }
    allocations = [
        device_ids[i] for i in
        sorted(list(range(len(device_ids))) * math.ceil(num_layers / len(device_ids)))
    ]
    for layer_i, device_id in enumerate(allocations):
        device_map[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.down_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.up_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.input_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = device_id
    model = MultiLLaMAForCausalLM(
        lang_model_path=os.path.join(FOLDER_PATH, 'Language_files'),
    )
    print('Device Map:', device_map)
    try:
        ckpt = torch.load(os.path.join(FOLDER_PATH, 'pytorch_model.bin'), map_location='cpu') # Please dowloud our checkpoint from huggingface and Decompress the original zip file first
        model.load_state_dict(ckpt, strict=False)
    except RuntimeError as e:
        print("Caught an error during model loading:")
        print(e)
        # Check memory after failed load attempt (if possible)
        print("Memory usage after failed loading attempt:")
        for device in device_ids:
            torch.cuda.set_device(device)
            print_memory_usage(device)
        raise e

    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")
    print("Finish loading model")
    
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.module.eval() 
    all_generated_texts = []
    device = next(model.module.parameters()).device
    print("Model Running on device: ",device)
    with torch.no_grad():
        for file in tqdm(img_paths):
            image =[
                {
                    'img_path': file,
                    'position': 0, #indicate where to put the images in the text string, range from [0,len(question)-1]
                }, # can add abitrary number of imgs
            ] 

            text,vision_x = combine_and_preprocess(question,image,image_padding_tokens)
        
            lang_x = text_tokenizer(
                    text, max_length=2048, truncation=True, return_tensors="pt"
            )['input_ids'].to(device)
        
            vision_x = vision_x.to(device)
            generation = model.module.generate(lang_x,vision_x)
            generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
            all_generated_texts.append(generated_texts)
        
    return all_generated_texts

       