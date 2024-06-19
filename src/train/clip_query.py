from models import CLIPModel, SigLIPModel, SigLIPClassifierModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertModel, BertTokenizer
import json
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
query = input('input query: ')
encoded_query = tokenizer([query])
batch = {
    key: torch.tensor(values).to(device)
    for key, values in encoded_query.items()
}

model = SigLIPClassifierModel(trainable=False).to(device)
statedict = torch.load('/data/cl2920/Trained_Models/siglip_20emb_5.pt')
model.load_state_dict(statedict)
model.eval()

with torch.no_grad():
    text_features = model.text_encoder(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    text_embeddings = model.text_projection(text_features)
print(f'Similarity Matrix: {text_embeddings[:20]@text_embeddings[:20].T}')
    
with open('src/train/utils/embs_14_siglip_classification.json', 'r') as file:
    data = json.load(file)
    
image_embeddings = torch.tensor(data['embeddings']).to(device)

t = torch.exp(model.t_p[0])
image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1).squeeze(1)
text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
print("Text embeddings shape:", text_embeddings_n.shape)
print("Image embeddings shape:", image_embeddings_n.shape)
dot_similarity = image_embeddings_n @ text_embeddings_n.T * t + model.b[0]
scores = dot_similarity.squeeze(1)
print(scores[:20])
# multiplying by 5 to consider that there are 5 captions for a single image
# so in indices, the first 5 indices point to a single image, the second 5 indices
# to another one and so on.
values, indices = torch.topk(scores, 10)
print(f'top 10 values: {values}')
matches = [data['files'][idx] for idx in indices]

for match in matches:
    print(f'\'{match}\',')