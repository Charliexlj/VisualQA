import torch
from torch.utils.data import DataLoader
from models import CLIPModel, SigLIPModel, SigLIPClassifierModel, CLIPDataset, Vision_Test_Model
import json
import argparse
from tqdm import tqdm
import numpy as np

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--full-param', type=str2bool, required=False, default=False)
parser.add_argument('--weight', type=str, required=False, default=None)
parser.add_argument('--data', type=str, required=True, default=None)
parser.add_argument('--save', type=str, required=True, default=None)
parser.add_argument('--batch-size', type=int, required=False, default=16)
parser.add_argument('--subset', type=str2bool, required=False, default=False)
parser.add_argument('--num-eps', type=int, required=False, default=10) 
parser.add_argument('--panelty', type=str2bool, required=False, default=False)
parser.add_argument('--model', type=str, required=False, default='siglip')
parser.add_argument('--alpha', type=int, required=False, default=5)
parser.add_argument('--beta', type=int, required=False, default=5)
parser.add_argument('--base', type=int, required=False, default=10)
parser.add_argument('--save-per-epoch', type=int, required=False, default=10)
args = parser.parse_args()

print('Current Training Configuration:')
for arg in vars(args):
    print(f'{arg}: {getattr(args, arg)}')

# Load the JSON data
with open(args.data, 'r') as file:
    raw_data = json.load(file)

if args.subset:
    raw_data = {k: v[:20] for k, v in raw_data.items()}

DATA_DIR = '/data/cl2920/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset and DataLoader preparation
train_dataset = CLIPDataset(
    image_files=[DATA_DIR + file for file in raw_data['files']],
    tokenized_reports=raw_data['tokenized_concise_reports'],
    labels = None if args.model == 'siglip' else raw_data['labels'],
    device=device
)
train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

# Model setup with DataParallel
if args.model == 'siglip_classifier':
    model = SigLIPClassifierModel(trainable=args.full_param).to(device)
elif args.model == 'siglip':
    model = SigLIPModel(trainable=args.full_param).to(device)
elif args.model == 'clip':
    model = CLIPModel(trainable=args.full_param).to(device)
else:
    model = Vision_Test_Model(trainable=args.full_param).to(device)
    
    
if args.weight:
    statedict = torch.load(args.weight)
    model.load_state_dict(statedict, strict=False)
model = torch.nn.DataParallel(model, device_ids=[0, 1])  # Utilizing both GPUs

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# for p in model.module.text_encoder.base.parameters():
#     p.requires_grad = False
# for p in model.module.text_projection.parameters():
#     p.requires_grad = False
# print('Text Encoder Frozen')
def train(model, train_dataloaders, optimizer, num_epochs=10, save_path=args.save):
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_dataloaders), total=len(train_dataloaders), leave=True)
        if epoch == 1:
            check_state = [p.requires_grad for p in model.module.text_encoder.base.parameters()].extend(
                [p.requires_grad for p in model.module.text_projection.parameters()]
            )
            print('Check if Text Encoder is Frozen:', np.unique(check_state))
        for i, sample in loop:
            image_tensor, input_ids, attention_mask, labels = sample
            image_tensor = image_tensor.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            if args.model == 'siglip':
                loss, (zimg_norm_diff, ztxt_norm_diff) = model(image_tensor, input_ids, attention_mask)
                if loss.dim() > 0:  # Check if loss is not a scalar
                    loss = loss.mean()
                zimg_norm_diff = zimg_norm_diff.mean()
                ztxt_norm_diff = ztxt_norm_diff.mean()
                total_loss = loss + torch.abs(1-zimg_norm_diff) + torch.abs(1-ztxt_norm_diff)
                if args.panelty:
                    total_loss.backward()
                else:
                    loss.backward()
                loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
                loop.set_postfix(loss=loss.item(), total_loss=total_loss.item(), zimg_norm_diff=zimg_norm_diff.mean().item(), ztxt_norm_diff=ztxt_norm_diff.mean().item())
            else:
                if args.model=='vision_test':
                    loss, auc = model(image_tensor, labels)
                    if loss.dim() > 0:  # Check if loss is not a scalar
                        loss = loss.mean()
                    loss.backward()
                    loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
                    loop.set_postfix(loss=loss.item(), auc=auc.mean().item())
                if args.model=='siglip_classifier':
                    total_loss, (sim_loss, img_class_loss, txt_class_loss), (img_auc, txt_auc) = model(
                        image_tensor, input_ids, attention_mask, labels,
                        alpha=args.alpha/args.base, beta=args.beta/args.base)
                    while total_loss.dim() > 0:
                        total_loss = total_loss.mean()
                    total_loss.backward()
                    loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
                    loop.set_postfix(
                        loss=f"{total_loss.item():.2f}",
                        sim=f"{sim_loss.mean().item():.2f}",
                        img_class=f"{img_class_loss.mean().item():.2f}",
                        txt_class=f"{txt_class_loss.mean().item():.2f}",
                        img_auc=f"{img_auc.mean().item():.2f}",
                        txt_auc=f"{txt_auc.mean().item():.2f}"
                    )            
            optimizer.step()

        if (epoch+1) % args.save_per_epoch == 0:
            dict_path = f'{save_path}_{epoch+1}.pt'
            torch.save(model.module.state_dict(), dict_path)
            print(f'Model saved to {dict_path}')

train(model, train_dataloaders, optimizer, num_epochs=args.num_eps, save_path=args.save)