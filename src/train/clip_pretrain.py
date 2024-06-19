from models import CLIPModel, CLIPDataset
import torch
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--full-param', type=bool, required=False, default=False)
parser.add_argument('--weight', type=str, required=False, default=None)
parser.add_argument('--data', type=str, required=True, default=None)
parser.add_argument('--save', type=str, required=True, default=None)
parser.add_argument('--batch-size', type=int, required=False, default=16)
parser.add_argument('--num-eps', type=int, required=False, default=10)
args = parser.parse_args()

# Load the JSON file
with open(args.data, 'r') as file:
    raw_data = json.load(file)
    
DATA_DIR = '/data/cl2920/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = CLIPDataset(
    image_files=[DATA_DIR + file for file in raw_data['files']],
    tokenized_reports=raw_data['tokenized_concise_reports'],
    device=device
)

print(f'max_len: {train_dataset.max_len}')

train_dataloaders = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True)

model = CLIPModel(trainable=args.full_param).to(device)
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_trainable_params}")

if args.weight:
    statedict = torch.load(args.weight)
    model.load_state_dict(statedict)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

def train(model,
          train_dataloaders,
          optimizer,
          num_epochs=10,
          save_path=args.save):
    for epoch in range(num_epochs):
        # Initialize tqdm loop
        loop = tqdm(enumerate(train_dataloaders), total=len(train_dataloaders), leave=True)
        for i, sample in loop:
            image_tensor, input_ids, attention_mask = sample
            image_tensor = image_tensor.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            loss = model(image_tensor, input_ids)
            loss.backward()
            optimizer.step()
            
            loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
            loop.set_postfix(loss=loss.item())

    # Save the model after training
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    
train(model, train_dataloaders, optimizer, num_epochs=args.num_eps, save_path=args.save)