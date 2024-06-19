import json
import os
from evaluate import load
import pandas as pd

# Load data from JSON file
json_filename = "output/vlm/impressions.json"
with open(json_filename, 'r') as json_file:
    data_loaded = json.load(json_file)

# Ensure y_label_list is a list of lists of strings
y_label_list = [[item] for item in data_loaded["impressions_label"]]
y_pred_list = [[item] for item in data_loaded["impressions_pred"]]

bleu = load("bleu", trust_remote_code=True)
rouge = load("rouge", trust_remote_code=True)
meteor = load("meteor", trust_remote_code=True)

# Compute BLEU, ROUGE, and METEOR scores
all_results = []
for count, pred in enumerate(y_pred_list):
    results = {}
    # Computing BLEU-1 to BLEU-4
    for j in range(1, 5):
        results[f'BLEU-{j}'] = bleu.compute(predictions=pred, references=y_label_list[count], max_order=j)["bleu"]
    # Computing ROUGE scores
    rouge_result = rouge.compute(predictions=pred, references=y_label_list[count])
    results.update({
        'ROUGE-1': rouge_result['rouge1'],
        'ROUGE-2': rouge_result['rouge2'],
        'ROUGE-L': rouge_result['rougeL']
    })
    # Computing METEOR score
    meteor_result = meteor.compute(predictions=pred, references=y_label_list[count])
    results['METEOR'] = meteor_result['meteor']
    all_results.append(results)
    if (count+1) % 100 == 0:
        print(f'processed {count+1} labels')

# Save results to JSON file
json_filename = "./output/eval/bleu_rouge_meteor.json" # TODO: Modify the path to save the results
with open(json_filename, 'w') as json_file:
    json.dump(all_results, json_file)