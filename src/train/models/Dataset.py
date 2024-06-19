import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import torchxrayvision as xrv
import skimage, torch, torchvision
import sklearn.metrics as metrics

class CLIPDataset(Dataset):
    def __init__(self, image_files, tokenized_reports, device, labels=None):
        """
        Args:
            image_files (list of str): Filenames of images.
            directory (str): Path to the directory containing images.
            device (torch.device): The device to load tensors to.
        """
        self.image_files = image_files
        self.device = device
        self.labels = labels
        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        self.tokenized_reports = tokenized_reports
        max_len = max([len(report['input_ids'][0]) for report in tokenized_reports])
        self.max_len = max_len if max_len < 512 else 512
        
    def padding(self, sequence, pad_token_id):
        actual_len = len(sequence)
        if actual_len >= self.max_len:
            sequence = sequence[:self.max_len]

        padded_sequence = torch.full((self.max_len,), pad_token_id, dtype=sequence.dtype)
        padded_sequence[:actual_len] = sequence
        
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        attention_mask[:actual_len] = 1
        
        return padded_sequence, attention_mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = skimage.io.imread(self.image_files[idx])

        # Normalize and check image dimensions
        img = xrv.datasets.normalize(img, 255)
        if len(img.shape) > 2:
            img = img.mean(2)  # Convert to grayscale by averaging channels
        if len(img.shape) < 2:
            raise ValueError("Error, dimension lower than 2 for image")

        # Add channel dimension
        img = img[None, :, :]

        # Apply transformations
        img = self.transform(img)
        img_tensor = torch.from_numpy(img).float().to(self.device)
        
        input_ids = torch.tensor(self.tokenized_reports[idx]['input_ids'][0]).to(self.device)
        input_ids, attention_mask = self.padding(input_ids, pad_token_id=0)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx]).to(self.device)
            # For MIMIC-CXR dataset, we need to convert the labels to binary
            # 0: Blank, 1:Positive, 2:Negative, 3:Uncertain
            # Conver Blank and Uncertain to torch.nan
            # if label[13] == 1 then the patient is health, so all other labels should be 2
            if label[13] == 1:
                label[:13] = 2
                
            label = label.float()
            label[torch.logical_or(label == 0, label == 3)] = torch.nan
            label[label == 2] = 0
            
            return img_tensor, input_ids, attention_mask, label

        return img_tensor, input_ids, attention_mask