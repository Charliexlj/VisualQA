from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import json
import numpy as np
import pandas as pd

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']

Critical_idx = [1, 4, 5, 7, 9]

with open("output/vlm_2/labels_and_predictions.json") as f:
    data = json.load(f)

# if True:#mimic-unfiltered
#     with open("src/train_v2/utils/cleaned_reports.json") as f:
#         data2 = json.load(f)

#     files = data2["files"]
#     files = [file[30:] for file in files]

#     df = pd.read_csv("local_test/mimic_test.csv")
#     paths = df["image_path"].tolist()

#     indices = [paths.index(file) for file in files]
    
#     data['y_pred'] = [data['y_pred'][i] for i in indices]  
    
if True: #NIH
    df = pd.read_csv('NIH_test.csv')
    label = [df['Cardiomegaly'].tolist(),
                df['Edema'].tolist(),
                df['Consolidation'].tolist(),
                df['Pneumonia'].tolist(),
                df['Atelectasis'].tolist(),
                df['Pneumothorax'].tolist(),
                df['Effusion'].tolist(),
                df['No Finding'].tolist(),]

    label = np.array(label)
    label = label.T
    
y_label = data["y_label"]
y_pred = data["y_pred"]
print((len(y_label), len(y_pred)))

y_label = np.array(y_label)
mask = y_label[:, 13] == 1
y_label[mask, :13] = 2
y_label = y_label.astype(float)
y_label[np.logical_or(y_label == 0, y_label == 3)] = np.nan
y_label[y_label == 2] = 0

y_pred = np.array(y_pred)
y_pred = y_pred.astype(float)
y_pred[np.logical_or(y_pred == 0, y_pred == 3)] = 0
y_pred[y_pred == 2] = 0

if True: #NIH
    NIH_IDX = [1, 4, 5, 6, 7, 8, 9, 13]
    Critical_idx = [0, 1, 2, 4, 6]
    df = pd.read_csv('NIH_test.csv')
    y_label = [df['Cardiomegaly'].tolist(),
                df['Edema'].tolist(),
                df['Consolidation'].tolist(),
                df['Pneumonia'].tolist(),
                df['Atelectasis'].tolist(),
                df['Pneumothorax'].tolist(),
                df['Effusion'].tolist(),
                df['No Finding'].tolist(),]

    y_label = np.array(y_label)
    y_label = y_label.T
    print(y_label.shape, y_pred.shape)
    
    y_pred = [y_pred[:,i] for i in NIH_IDX]
    y_pred = np.array(y_pred).T
    
    print(y_label.shape, y_pred.shape)

mask = ~np.isnan(y_label)


all_auc = []
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []

auc_txt = ''
acc_txt = ''
precision_txt = ''
recall_txt = ''
f1_txt = ''

critical_f1 = []
critical_acc = []
critical_precision = []
critical_recall = []

for i in range(8):
    # Apply the mask to the labels and predictions
    true_labels = y_label[:, i][mask[:, i]]
    predictions = y_pred[:, i][mask[:, i]]

    # Calculate AUC
    # auc = roc_auc_score(true_labels, predictions)
    # all_auc.append(auc)
    # auc_txt += f'{CONDITIONS[i]} AUC: {auc:.4f}\n'

    # Calculate accuracy - assuming your predictions are probability scores, you might need to threshold them
    predicted_labels = predictions > 0.5
    accuracy = accuracy_score(true_labels, predicted_labels)
    all_accuracy.append(accuracy)
    acc_txt += f'{CONDITIONS[i]} Accuracy: {accuracy:.4f}\n'

    # Calculate precision
    precision = precision_score(true_labels, predicted_labels)
    all_precision.append(precision)
    precision_txt += f'{CONDITIONS[i]} Precision: {precision:.4f}\n'

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels)
    all_recall.append(recall)
    recall_txt += f'{CONDITIONS[i]} Recall: {recall:.4f}\n'

    # Calculate F1-score
    f1 = f1_score(true_labels, predicted_labels)
    all_f1.append(f1)
    f1_txt += f'{CONDITIONS[i]} F1-Score: {f1:.4f}\n'
    
    if i in Critical_idx:
        critical_f1.append(f1)
        critical_acc.append(accuracy)
        critical_precision.append(precision)
        critical_recall.append(recall)

# Calculate and print mean metrics
mean_auc = np.mean(all_auc)
mean_accuracy = np.mean(all_accuracy)
mean_precision = np.mean(all_precision)
mean_recall = np.mean(all_recall)
mean_f1 = np.mean(all_f1)

# print(f'Mean AUC: {mean_auc}')
# print(auc_txt)
# print(f'Mean Accuracy: {mean_accuracy}')
# print(acc_txt)
# print(f'Mean Precision: {mean_precision}')
# print(precision_txt)
# print(f'Mean Recall: {mean_recall}')
# print(recall_txt)
# print(f'Mean F1-Score: {mean_f1}')
# print(f1_txt)

# print(f'Mean AUC: {mean_auc}')
print(f'Mean Accuracy: {mean_accuracy}')
print(f'Mean Precision: {mean_precision}')
print(f'Mean Recall: {mean_recall}')
print(f'Mean F1-Score: {mean_f1}')
print()
print(f'Critical Accuracy: {np.mean(critical_acc)}')
print(f'Critical Precision: {np.mean(critical_precision)}')
print(f'Critical Recall: {np.mean(critical_recall)}')
print(f'Critical F1-Score: {np.mean(critical_f1)}')
