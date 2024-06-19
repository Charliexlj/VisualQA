import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
sys.path.append('/homes/cl2920/Projects/CheXmed-VisualQA/src/train_v2')

from adaptors import MIMIC_Findings, CheXbert_label, XrayGPT_Findings , LlavaMed_Findings, VLM_Findings
import yaml

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']
CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}

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

def label_to_text(label):
    try:
        out = {"Positive": [CONDITIONS[i] for i, l in enumerate(label) if l == 1],
            "Negative": [CONDITIONS[i] for i, l in enumerate(label) if l == 2],
            "Uncertain": [CONDITIONS[i] for i, l in enumerate(label) if l == 3]}
    except IndexError:
        out = {"Positive": [],
            "Negative": [],
            "Uncertain": []}
        print(f'Len of Conditions: {len(CONDITIONS)}', f'Len of Label: {len(label)}')
    return out

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

if __name__ == '__main__':
    ############################################################################################################
    # Label the ground truth impressions using CheXbert
    # impressions_label: List of impressions from the ground truth example: ["The chest x-ray findings reveal ...", ...]
    # y_label: Numpy array of shape (num_samples, num_conditions) with values 0, 1, 2, 3
    ############################################################################################################
    if config["dataset"] == "mimic":
    # Load the dataset
        dataset = pd.read_csv(config["dataset_directory"] + "/mimic-cxr-2.0.0-split.csv.gz",
                            dtype={'study_id': str, 'subject_id': str}).query('split == "test"')
        
        # If config["num_samples"] is "all", use all samples; otherwise, sample n records
        if config["num_samples"] == "all":
            test_samples = dataset[['study_id', 'subject_id']].to_dict('records')
        else:
            test_samples = dataset.sample(n=config["num_samples"])[['study_id', 'subject_id']].to_dict('records')
            
        print(f"Testing on number of samples: {len(test_samples)}")
                        
        files = ['/files/p' + sample['subject_id'][:2] + '/p' +  sample['subject_id'] + '/s' + sample['study_id'] for sample in test_samples]
        
        reports_data = [{"Report Impression": MIMIC_Findings(config["dataset_directory"] + file + ".txt")} for file in files]
        impressions_label = collect_impressions(reports_data)
        
        y_label = CheXbert_label(config["checkpoint_path"], impressions_label)
        y_label = np.array(y_label).T
        
    elif config["dataset"] == "mimic_APPA":
        # Load the dataset
        # DicomPath,Reports,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Lesion,Lung Opacity,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices
        dataset = pd.read_csv(config["dataset_directory"] + "MIMIC_AP_PA_test.csv")
                            # dtype={'DicomPath': str, 'Reports': str, 'No Finding': 'Int64', 'Enlarged Cardiomediastinum': 'Int64', 'Cardiomegaly': 'Int64', 'Lung Lesion': 'Int64', 'Lung Opacity': 'Int64', 'Edema': 'Int64', 'Consolidation': 'Int64', 'Pneumonia': 'Int64', 'Atelectasis': 'Int64', 'Pneumothorax': 'Int64', 'Pleural Effusion': 'Int64', 'Pleural Other': 'Int64', 'Fracture': 'Int64', 'Support Devices': 'Int64'})
        
        # If config["num_samples"] is "all", use all samples; otherwise, sample n records
        if config["num_samples"] == "all":
            test_samples = dataset.to_dict('records')
        else:
            test_samples = dataset.sample(n=config["num_samples"]).to_dict('records')
            
        print(f"Testing on number of samples: {len(test_samples)}")
                        
        files = [sample['DicomPath'] for sample in test_samples]
        
        impressions_label = [str(sample['Reports']) for sample in test_samples]
        
        y_label = np.array([[sample['Enlarged Cardiomediastinum'], sample['Cardiomegaly'], sample['Lung Lesion'], sample['Lung Opacity'], sample['Edema'], sample['Consolidation'], sample['Pneumonia'], sample['Atelectasis'], sample['Pneumothorax'], sample['Pleural Effusion'], sample['Pleural Other'], sample['Fracture'], sample['Support Devices'], sample['No Finding']] for sample in test_samples])

        # Apply the mapping to the entire array
        y_label = vectorized_apply_mapping(y_label)
        
    elif config["dataset"] == "nih-cxr":
        dataset = pd.read_csv(config["dataset_directory"] + "../test_list.csv")
        if config["num_samples"] == "all":
            test_samples = dataset.to_dict('records')
        else:
            test_samples = dataset.sample(n=config["num_samples"]).to_dict('records')
        print(f"Testing on number of samples: {len(test_samples)}")
                        
        files = [sample['Image Index'] for sample in test_samples]
        
        y_label = [[0 if s['No Finding'] else np.nan,
                   s['Cardiomegaly'],
                   s['Infiltration'],
                   1 if s['Mass']==1 or s['Nodule']==1 else 0,
                   s['Edema'],
                   s['Consolidation'],
                   s['Pneumonia'],
                   s['Atelectasis'],
                   s['Pneumothorax'],
                   s['Effusion'],
                   s['Pleural_Thickening'],
                   0 if s['No Finding'] else np.nan,
                   0 if s['No Finding'] else np.nan,
                   s['No Finding']] for s in test_samples]
        y_label = np.array(y_label)
        y_label = vectorized_apply_mapping(y_label)
        print('shape of y_label:', y_label.shape)
        impressions_label = [] # placeholder
        
    else:
        raise ValueError(f"Dataset {config['dataset']} not supported")
    
    
    ############################################################################################################
    # Get the predictions from the model
    # impressions_pred: List of impressions from the model: ["The chest x-ray findings reveal ...", ...]
    # y_pred: Numpy array of shape (num_samples, num_conditions) with values 0, 1, 2, 3
    ############################################################################################################
    if config["model"] == "xraygpt":
        model_answer_data = XrayGPT_Findings([config["dataset_directory"] + file for file in files], config["model_path"]+"/eval_configs/xraygpt_eval.yaml", config['gpu_id'], direct_path=True)    
        model_answer_data = [{"Report Impression": findings} for findings in model_answer_data]
        impressions_pred = collect_impressions(model_answer_data)
        
        y_pred = CheXbert_label(config["checkpoint_path"], impressions_pred)
        y_pred = np.array(y_pred).T
        
    elif config["model"] == "llava_med" or config["model"] == "llava_med_QA":
        print(config["model"])
        model_answer_data = LlavaMed_Findings(config["model_path"], [config["dataset_directory"] + file for file in files])    
        model_answer_data = [{"Report Impression": findings} for findings in model_answer_data]
        impressions_pred = collect_impressions(model_answer_data)
        
        y_pred = CheXbert_label(config["checkpoint_path"], impressions_pred)
        y_pred = np.array(y_pred).T

    # elif config["model"] == "radfm":
    #     model_answer_data = RadFM_Findings([config["dataset_directory"] + file for file in files], config["model_path"])
    #     model_answer_data = [{"Report Impression": findings} for findings in model_answer_data]
    #     impressions_pred = collect_impressions(model_answer_data)
        
    #     y_pred = CheXbert_label(config["checkpoint_path"], impressions_pred)
    #     y_pred = np.array(y_pred).T
    
    elif config["model"] == "radfm_3":
        radfm_data_path = "output/radfm/RadFM_nih-cxr.csv"
        radfm_df = pd.read_csv(radfm_data_path)
        
        model_answer_data = radfm_df['Pred'].tolist()
        model_answer_data = [{"Report Impression": findings} for findings in model_answer_data]
        impressions_pred = collect_impressions(model_answer_data)
        
        y_pred = CheXbert_label(config["checkpoint_path"], impressions_pred)
        y_pred = np.array(y_pred).T
        
    elif config["model"] == "vlm":
        model_answer_data = VLM_Findings([config["dataset_directory"] + file for file in files])
        model_answer_data = [{"Report Impression": findings} for findings in model_answer_data]
        impressions_pred = collect_impressions(model_answer_data)
        
        y_pred = CheXbert_label(config["checkpoint_path"], impressions_pred)
        y_pred = np.array(y_pred).T
    else:
        raise ValueError(f"Model {config['model']} not supported")
    ############################################################################################################
    # Save the results
    ############################################################################################################
    collective_output = ""
    for i, f in enumerate(files):
        text_label = label_to_text(y_label[i])
        text_pred = label_to_text(y_pred[i])
        output_text = f"File Number {i}: {f}\nLabel:\nPositive: {text_label['Positive']}\nNegative: {text_label['Negative']}\nUncertain: {text_label['Uncertain']}\n\n{impressions_label[i] if len(impressions_label)>0 else ''}" + \
              f"\n\nPrediction:\nPositive: {text_pred['Positive']}\nNegative: {text_pred['Negative']}\nUncertain: {text_pred['Uncertain']}\n\n{impressions_pred[i]}\n\n\n"
        collective_output += output_text
    # print(collective_output)
    
    base_dir = "./output"
    model_dir = os.path.join(base_dir, config["model"])
    os.makedirs(model_dir, exist_ok=True)

    txt_filename = os.path.join(model_dir, "summery.txt")

    with open(txt_filename, 'w') as file:
        file.write(collective_output)
    print(f"Report Data saved to {txt_filename}")

    impression_to_save = {
        "impressions_label": impressions_label,
        "impressions_pred": impressions_pred.tolist(),
    }
    json_filename = os.path.join(model_dir, "impressions.json")
    with open(json_filename, 'w') as json_file:
        json.dump(impression_to_save, json_file, indent=4)

    # Data for labels_and_predictions.json
    label_to_save = {
        "y_label": y_label.tolist(),
        "y_pred": y_pred.tolist(),
    }

    # Saving the labels_and_predictions.json file
    json_filename = os.path.join(model_dir, "labels_and_predictions_5000.json")

    with open(json_filename, 'w') as json_file:
        json.dump(label_to_save, json_file, indent=4)

    print(f"Label Data saved to {json_filename}")
