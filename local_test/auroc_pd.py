from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import numpy as np
import pandas as pd

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']

Critical_idx = [1, 4, 5, 7, 9]
# Critical_CONS = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    
if True: #NIH
    df = pd.read_csv('NIH_test.csv')
    image_index = df['Image Index'].tolist()
    
#######################
# CHANGE THIS TO CORRESPONDING FILE
#######################
with open("visualqa_labels_and_predictions.json") as f:
    data = json.load(f)
    
y_pred = data["y_pred"]

# convert blank and uncertain to negative, now only 0 and 1
# 0 for negative, 1 for positive
y_pred = np.array(y_pred)
y_pred = y_pred.astype(float)
y_pred[np.logical_or(y_pred == 0, y_pred == 3)] = 0
y_pred[y_pred == 2] = 0

if True: #NIH
    df = pd.read_csv('test_labeled.csv')
    y_label = df[CONDITIONS].values
    y_label = np.array(y_label)
    # y_label = y_label.T # idk? might need to transpose
    print('Shape of y_label and y_pred, should be (N, 14):')
    print(y_label.shape, y_pred.shape) # here should be (N, 14) and (N, 14)
    
    label_image_index = df['image_1'].tolist()
    index_map = {value: idx for idx, value in enumerate(label_image_index)}
    idx_order = [index_map[value] for value in image_index]
    y_label = y_label[idx_order] # reorder y_label to match y_pred


y_label = np.array(y_label)
mask = y_label[:, 13] == 1 # if no finding
y_label[mask, :13] = 0 # set all other conditions to negative
y_label = y_label.astype(float)
y_label[y_label == -1] = np.nan # set other uncertain to nan

# y_pred is my own label, 0:blank, 1:positive, 2:negative, 3:uncertain
y_pred = np.array(y_pred)
y_pred = y_pred.astype(float)
y_pred[np.logical_or(y_pred == 0, y_pred == 3)] = 0
y_pred[y_pred == 2] = 0

mask = ~np.isnan(y_label)

all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []

acc_txt = ''
precision_txt = ''
recall_txt = ''
f1_txt = ''

critical_f1 = []
critical_acc = []
critical_precision = []
critical_recall = []

for i in range(y_label.shape[1]):
    # Apply the mask to the labels and predictions
    true_labels = y_label[:, i][mask[:, i]]
    predictions = y_pred[:, i][mask[:, i]]

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

mean_accuracy = np.mean(all_accuracy)
mean_precision = np.mean(all_precision)
mean_recall = np.mean(all_recall)
mean_f1 = np.mean(all_f1)

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
