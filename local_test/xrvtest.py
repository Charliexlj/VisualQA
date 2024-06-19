from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import json
import numpy as np

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']

AVAIL_CONS = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Effusion",
        "Fracture"
    ]

vision_ppv80 = [
    0.70,  # Enlarged Cardiomediastinum
    0.888,  # Cardiomegaly
    0.85,  # Lung Opacity
    0.65,  # Lung Lesion
    0.65,  # Edema
    0.925,  # Consolidation
    0.62,  # Pneumonia
    0.72,  # Atelectasis
    0.849,  # Pneumothorax
    0.687,  # Effusion
    0.85,  # Fracture
]

with open("output/torchvision/results.json") as f:
    data = json.load(f)
    
y_label = data["label"]
y_pred = data["pred"]

y_label = np.array(y_label)
mask = y_label[:, 13] == 1
y_label[mask, :13] = 2
y_label = y_label.astype(float)
y_label[np.logical_or(y_label == 0, y_label == 3)] = np.nan
y_label[y_label == 2] = 0

y_label = np.hstack((y_label[:, :10], y_label[:, 11:12]))

print(np.unique(y_label))

y_pred = np.array(y_pred)
y_pred = y_pred.astype(float)
# y_pred[np.logical_or(y_pred == 0, y_pred == 3)] = 0
# y_pred[y_pred == 2] = 0

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
thresholds_txt = ''

for i in range(10):
    # Apply the mask to the labels and predictions
    true_labels = y_label[:, i][mask[:, i]]
    predictions = y_pred[:, i][mask[:, i]]

    # Calculate AUC
    auc = roc_auc_score(true_labels, predictions)
    all_auc.append(auc)
    auc_txt += f'{CONDITIONS[i]} AUC: {auc:.4f}\n'
    
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    # print(('condition:', CONDITIONS[i]))
    # print('fpr:', fpr)
    # print('tpr:', tpr)
    # print('thresholds:', thresholds)

    # Find the optimal threshold: the one that minimizes the distance to the top-left corner
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    thresholds_txt += f'{CONDITIONS[i]} Optimal Threshold: {optimal_threshold:.4f}\n'

    # Calculate accuracy - assuming your predictions are probability scores, you might need to threshold them
    predicted_labels = predictions > optimal_threshold
    # predicted_labels = predictions > vision_ppv80[i]
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
# print(f'Mean Accuracy: {mean_accuracy}')
# print(f'Mean Precision: {mean_precision}')
# print(f'Mean Recall: {mean_recall}')
# print(f'Mean F1-Score: {mean_f1}')

print(thresholds_txt)