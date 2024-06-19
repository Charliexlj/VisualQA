import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
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

CONS = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# Print the column names
print(df.columns.tolist())

