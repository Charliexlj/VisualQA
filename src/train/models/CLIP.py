import torchxrayvision as xrv
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertModel, BertTokenizer
import sklearn.metrics as metrics
import numpy as np

PROJECTION_DIM = 14
CLASS_COUNT = 14

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']

class Projection(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim = PROJECTION_DIM,
        dropout = 0.2
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class VisionEncoder(nn.Module): # Densenet
    def __init__(self, trainable=False, name="densenet121-res224-all") -> None:
        super().__init__()
        self.base = xrv.models.DenseNet(weights=name).features
        for p in self.base.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        out = self.base(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        return out


class TextEncoder(nn.Module):
    def __init__(self, trainable=False, name='dmis-lab/biobert-base-cased-v1.1') -> None:
        super().__init__()
        self.base = BertModel.from_pretrained(name)
        for p in self.base.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask=None):
        out = self.base(input_ids=input_ids,
                        attention_mask=attention_mask)[0]
        out = out[:, 0, :]  # get CLS token output
        return out

class CLIP(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        vision_embedding=1024,
        text_embedding=768,
        trainable=False,
        finetune=False
    ):
        super().__init__()
        self.vision_encoder = VisionEncoder(trainable=trainable)
        self.text_encoder = TextEncoder(trainable=trainable)
        self.vision_projection = Projection(embedding_dim=vision_embedding)
        self.text_projection = Projection(embedding_dim=text_embedding)
        self.temperature = temperature
        
    def pairwise_norm_mean(self, batch):
        pairwise_diff = batch[:, None, :] - batch[None, :, :]
        norms = torch.norm(pairwise_diff, dim=-1)
        mean_norm = torch.mean(norms)
        return mean_norm

    def forward(self, image_tensor, input_ids, attention_mask=None):
        # Getting Image and Text Features
        image_features = self.vision_encoder(image_tensor)
        text_features = self.text_encoder(input_ids, attention_mask)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.vision_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = F.cross_entropy(logits, targets, reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        # Check model collaspe
        zimg_avg_norm = self.pairwise_norm_mean(image_embeddings)
        ztxt_avg_norm = self.pairwise_norm_mean(text_embeddings)
        return loss.mean(), (zimg_avg_norm, ztxt_avg_norm)
    
class SigLIP(CLIP):
    def __init__(
        self,
        temperature=0.1,
        vision_embedding=1024,
        text_embedding=768,
        trainable=False,
        finetune=False
    ):
        super().__init__(
            temperature=temperature,
            vision_embedding=vision_embedding,
            text_embedding=text_embedding,
            trainable=trainable,
            finetune=finetune
        )
        self.t_p = nn.Parameter(torch.tensor([temperature]))
        self.b = nn.Parameter(torch.tensor([0.0]))
        
    def pairwise_norm_mean(self, batch):
        pairwise_diff = batch[:, None, :] - batch[None, :, :]
        norms = torch.norm(pairwise_diff, dim=-1)
        mean_norm = torch.mean(norms)
        return mean_norm
    
    def forward(self, image_tensor, input_ids, attention_mask=None):
        # Getting Image and Text Features
        image_features = self.vision_encoder(image_tensor)
        text_features = self.text_encoder(input_ids, attention_mask)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.vision_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        # https://arxiv.org/pdf/2303.15343
        t = torch.exp(self.t_p[0])
        zimg = F.normalize(image_embeddings, p=2, dim=-1)
        ztxt = F.normalize(text_embeddings, p=2, dim=-1)
        logits = zimg @ ztxt.T * t + self.b[0]
        labels = 2*torch.eye(logits.shape[0]).to(logits.device) - torch.ones(logits.shape).to(logits.device)
        loss = -torch.sum(F.logsigmoid(labels*logits)) / logits.shape[0]
        
        # Check model collaspe
        zimg_avg_norm = self.pairwise_norm_mean(zimg)
        ztxt_avg_norm = self.pairwise_norm_mean(ztxt)
        return loss, (zimg_avg_norm, ztxt_avg_norm)
    
class SIGLIP_CLASSIFIER(CLIP):
    def __init__(
        self,
        temperature=0.1,
        vision_embedding=1024,
        text_embedding=768,
        trainable=False,
        finetune=False
    ):
        super().__init__(
            temperature=temperature,
            vision_embedding=vision_embedding,
            text_embedding=text_embedding,
            trainable=trainable,
            finetune=finetune
        )
        self.t_p = nn.Parameter(torch.tensor([temperature]))
        self.b = nn.Parameter(torch.tensor([0.0]))
        
        self.classifier = nn.Linear(PROJECTION_DIM, CLASS_COUNT)
        
    def calculate_auc(self, out, labels):
        task_outputs={}
        task_targets={}
        for task in range(labels.shape[1]):
            task_outputs[task] = []
            task_targets[task] = []

        for task in range(labels.shape[1]):
            task_output = out[:,task]
            task_target = labels[:,task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            task_outputs[task].append(task_output)
            task_targets[task].append(task_target)
            
        for task in range(len(task_targets)):
            task_outputs[task] = torch.cat(task_outputs[task])
            task_targets[task] = torch.cat(task_targets[task])
            
        task_aucs = []
        for task in range(len(task_targets)):
            if len(torch.unique(task_targets[task]))> 1:
                task_targets_np = task_targets[task].cpu().detach().numpy()
                task_outputs_np = task_outputs[task].cpu().detach().numpy()
                task_auc = metrics.roc_auc_score(task_targets_np, task_outputs_np)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)
                
        task_aucs = torch.tensor(task_aucs).to(labels.device)
        auc = torch.mean(task_aucs[~torch.isnan(task_aucs)])
        
        return auc
    
    def forward(self, image_tensor, input_ids, attention_mask=None, labels=None, alpha=0.5, beta=0.5): 
        # Getting Image and Text Features
        image_features = self.vision_encoder(image_tensor)
        text_features = self.text_encoder(input_ids, attention_mask)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.vision_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Similarity Loss
        # https://arxiv.org/pdf/2303.15343
        t = torch.exp(self.t_p[0])
        zimg = F.normalize(image_embeddings, p=2, dim=-1)
        ztxt = F.normalize(text_embeddings, p=2, dim=-1)
        logits = zimg @ ztxt.T * t + self.b[0]
        # Original Labels, oppose if i != j
        sim_labels = 2*logits.shape[0]*torch.eye(logits.shape[0]).to(logits.device) - torch.ones(logits.shape).to(logits.device)
        # sim_loss = -torch.sum(F.logsigmoid(sim_labels*logits)) / logits.shape[0]
        # Trial generate self similarity first, shape (batch_size, batch_size)
        img_sim = zimg @ zimg.T
        txt_sim = ztxt @ ztxt.T
        if logits.device == torch.device('cuda:0'):
            # print(f'Logits:{logits[:5, :5]}')
            # print(f'Similarity Matrix of image: {zimg[:5]@zimg[:5].T}')
            # print(f'Similarity Matrix of txt: {ztxt[:5]@ztxt[:5].T}')
            print
        sim_loss = (-torch.sum(F.logsigmoid(sim_labels*logits)) / logits.shape[0])# +(-torch.sum(F.logsigmoid(sim_labels*img_sim)) / img_sim.shape[0])+(-torch.sum(F.logsigmoid(sim_labels*txt_sim)) / txt_sim.shape[0])
        # sim_loss = (-torch.sum(F.logsigmoid(sim_labels*img_sim)) / img_sim.shape[0])
        
        # Calculating the Classification Loss
        mask = ~torch.isnan(labels)
        img_class_logits = image_embeddings# self.classifier(image_embeddings)
        img_class_loss = F.binary_cross_entropy_with_logits(
            img_class_logits[mask],
            labels[mask],
            reduction='sum')
        txt_class_logits = text_embeddings# self.classifier(text_embeddings)
        txt_class_loss = F.binary_cross_entropy_with_logits(
            txt_class_logits[mask],
            labels[mask],
            reduction='sum')
        class_loss = 0.5*(img_class_loss + txt_class_loss)
        
        # Total backpropagation loss
        total_loss = alpha*sim_loss  + beta*class_loss
        
        # AUC Calculation
        img_auc = self.calculate_auc(img_class_logits, labels)
        txt_auc = self.calculate_auc(txt_class_logits, labels)

        return class_loss, (sim_loss, img_class_loss, txt_class_loss), (img_auc, txt_auc)
    
    
    
    
    
class Vision_Test_Model(VisionEncoder):
    def __init__(self, trainable=False, name="densenet121-res224-all") -> None:
        super().__init__(trainable=trainable, name=name)
        self.classifier = nn.Linear(1024, CLASS_COUNT)
    
    def forward(self, x, labels):
        out = self.base(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = self.classifier(out)
        labels = labels.float()
        mask = ~torch.isnan(labels)
        loss = F.binary_cross_entropy_with_logits(out[mask], labels[mask])
        
        task_outputs={}
        task_targets={}
        for task in range(labels.shape[1]):
            task_outputs[task] = []
            task_targets[task] = []

        for task in range(labels.shape[1]):
            task_output = out[:,task]
            task_target = labels[:,task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            task_outputs[task].append(task_output)
            task_targets[task].append(task_target)
            
        for task in range(len(task_targets)):
            task_outputs[task] = torch.cat(task_outputs[task])
            task_targets[task] = torch.cat(task_targets[task])
            
        task_aucs = []
        for task in range(len(task_targets)):
            if len(torch.unique(task_targets[task]))> 1:
                task_targets_np = task_targets[task].cpu().detach().numpy()
                task_outputs_np = task_outputs[task].cpu().detach().numpy()
                task_auc = metrics.roc_auc_score(task_targets_np, task_outputs_np)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)
                

        task_aucs = torch.tensor(task_aucs).to(loss.device)
        for i, auc in enumerate(task_aucs):
            print(f'{CONDITIONS[i]} AUC = {auc.item():4.4f}')
        auc = torch.mean(task_aucs[~torch.isnan(task_aucs)])
        
        return loss, auc
