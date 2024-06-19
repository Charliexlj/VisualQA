from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

from adaptors import CheXbert_label
import yaml

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']
CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id,
    use_auth_token=...)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    use_auth_token=...
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

PROMPT_PRE = """Extract the most crucial findings from a radiology report in one sentense, focusing on disease identification and only include positive conditions with is provided, separate each short description with a semi-colon ';'. Remove all personal identifiers and irrelevant comparisons, make it as short as possible. For support devices, do not list the name of the devices, just summerise as support devices.

Example:
Raw Report: In comparison with the study of ___, there is no evidence of pneumothorax. \n Continued low lung volumes with substantial mass in the right paratracheal\n region.
Positive Conditions: Lung Lesion, Lung Opacity
Concise Version: substantial mass in paratracheal region; low lung volumes.

Raw Report: AP radiograph of the chest was reviewed in comparison to ___.\n \n The ET tube tip is 3.5 cm above the carina.  NG tube tip is in the stomach. \n There is left retrocardiac opacity, unchanged since the prior study.  Minimal\n interstitial pulmonary edema is unchanged.  No interval development of pleural\n effusion or pneumothorax is seen.
Positive Conditions: Lung Opacity, Edema, Support Devices
Concise Version: left retrocardiac opacity; minimal interstitial pulmonary edemapulmonary edema; support device in place.

Raw Report: The patient is markedly rotated to his left limiting evaluation of the cardiac\n and mediastinal contours.  The heart remains enlarged.  There has been\n interval removal of the endotracheal tube with placement of a tracheostomy\n tube, which has its tip at the thoracic inlet.  The right subclavian PICC line\n still has its tip in the distal SVC.  A nasogastric tube is seen coursing\n below the diaphragm with the tip projecting over the expected location in the\n stomach.  Patchy opacity in the retrocardiac region may reflect an area of\n atelectasis, although pneumonia cannot be entirely excluded.  No evidence of\n pulmonary edema.  No pneumothorax.  Probable small layering left effusion.
Positive Conditions: Enlarged Cardiomediastinum, Atelectasis, Pneumonia, Pleural Effusion, Support Devices
Concise Version: enlarged heart; possibly atelectasis or pneumonia; probable left pleural effusion; support devices in place.

Raw Report: As compared to the previous radiograph, there is marked improvement\n in extent and severity of the pre-existing parenchymal opacities.  Unchanged\n borderline size of the cardiac silhouette.  No pleural effusions.  The\n nasogastric tube has been removed.  Endotracheal tube and the right internal\n jugular vein introduction sheath are in constant position.
Positive Conditions: Lung Opacity, Support Devices
Concise Version: parenchymal opacities; support devices in place.

Task: Only give the concise version of the report. Do not include any other information in the response including "Concise Version:".
Raw Report: """

PROMPT_MID = """
Positive Conditions: """

PROMPT_POST = """
Concise Version: """

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

parser = argparse.ArgumentParser(description='Process the config file path.')
parser.add_argument('--eval-cfg', type=str, required=True,
                    help='Path to the configuration file')
args = parser.parse_args()

with open(args.eval_cfg, 'r') as file:
    config = yaml.safe_load(file)
    
def collect_impressions(reports_data):   
    df = pd.DataFrame(reports_data)
    imp = df['Report Impression']
    imp = imp.str.strip()
    imp = imp.replace('\n',' ', regex=True).replace('\s+', ' ', regex=True)
    imp = imp.str.strip()
    return imp

dataset = pd.read_csv('/data/cl2920/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/' + "MIMIC_AP_PA_train.csv")
                            # dtype={'DicomPath': str, 'Reports': str, 'No Finding': 'Int64', 'Enlarged Cardiomediastinum': 'Int64', 'Cardiomegaly': 'Int64', 'Lung Lesion': 'Int64', 'Lung Opacity': 'Int64', 'Edema': 'Int64', 'Consolidation': 'Int64', 'Pneumonia': 'Int64', 'Atelectasis': 'Int64', 'Pneumothorax': 'Int64', 'Pleural Effusion': 'Int64', 'Pleural Other': 'Int64', 'Fracture': 'Int64', 'Support Devices': 'Int64'})
        
# If config["num_samples"] is "all", use all samples; otherwise, sample n records
# if config["num_samples"] == "all":
test_samples = dataset.to_dict('records')
# else:
#     test_samples = dataset.sample(n=config["num_samples"]).to_dict('records')
    
print(f"Testing on number of samples: {len(test_samples)}")
                
files = [sample['DicomPath'] for sample in test_samples]

impressions_label = [str(sample['Reports']) for sample in test_samples]

y_label = np.array([[sample['Enlarged Cardiomediastinum'], sample['Cardiomegaly'], sample['Lung Lesion'], sample['Lung Opacity'], sample['Edema'], sample['Consolidation'], sample['Pneumonia'], sample['Atelectasis'], sample['Pneumothorax'], sample['Pleural Effusion'], sample['Pleural Other'], sample['Fracture'], sample['Support Devices'], sample['No Finding']] for sample in test_samples])
def apply_mapping(value):
    if pd.isna(value):  # Check if value is NaN
        return 0  # Blank
    elif value == 1:
        return 1  # Positive
    elif value == 0:
        return 2  # Negative
    elif value == -1:
        return 3  # Uncertain
    else:
        return value  # Just in case there are other values, handle appropriately

# Vectorize the apply_mapping function for efficiency
vectorized_apply_mapping = np.vectorize(apply_mapping)

# Apply the mapping to the entire array
y_label = vectorized_apply_mapping(y_label)

concise_reports = []

for idx, imp in enumerate(tqdm(impressions_label, desc="Processing Impressions")):
    messages = [
    {"role": "system", "content": "You are a radiology report editor, you will help to concise radiology reports as requested."},
    {"role": "user", "content": PROMPT_PRE + imp + PROMPT_MID + ", ".join([CONDITIONS[i] for i, x in enumerate(y_label[idx]) if x==1]) + PROMPT_POST},
    ]
    
    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
    ).to(model.device)
    
    attention_mask = torch.ones(input_ids.shape, dtype=torch.bfloat16).to(model.device)
    
    outputs = model.generate(
    input_ids,
    max_new_tokens=1024,
    eos_token_id=terminators,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=attention_mask,
    do_sample=True,
    temperature=0.1,
    top_p=0.1,
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    concise_reports.append({"Report Impression": tokenizer.decode(response, skip_special_tokens=True)})
    
concise_impressions = collect_impressions(concise_reports)
concise_label = CheXbert_label(config["checkpoint_path"], concise_impressions)
concise_label = np.array(concise_label).T

# Save y_label and concise_label as dictionaries
labels = {
    "files": files,
    "raw_reports": impressions_label,
    "concise_reports": [c["Report Impression"] for c in concise_reports],
    "raw_label": y_label.tolist(),
    "concise_label": concise_label.tolist(),
}

# Define the output file path
output_dir = "./output/llama_8B_instruct/"
output_file = os.path.join(output_dir, "new_labels.json")

# Check if the directory exists, if not create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the labels as JSON
with open(output_file, 'w') as file:
    json.dump(labels, file, indent=4)