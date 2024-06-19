import json

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']
CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}

with open('output/llama_8B_instruct/new_labels.json', 'r') as file:
    data = json.load(file)

all_files = data['files']
all_labels = data['raw_label']

files = [
...
]

file_indices = [all_files.index(file) for file in files]

idx_of_support_devices = CONDITIONS.index('Support Devices')

for i in file_indices:
    print(CLASS_MAPPING[all_labels[i][idx_of_support_devices]])
