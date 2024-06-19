import pandas as pd
import csv
from openai import OpenAI
from tqdm import tqdm
import json
import os
import numpy as np

df = pd.read_csv('src/train_v2/utils/data.csv')

# Assuming df contains your DataFrame
dicom_paths = df['DicomPath'].tolist()
reports = df['Reports'].tolist()
reports = [str(report) for report in reports]
print('Number of reports:', len(reports))
print('Number of dicom paths:', len(dicom_paths))

if input('Test a small portion of the data? (y/n)') == 'y':
    reports = reports[:10]
    dicom_paths = dicom_paths[:10]
SYS_MSG = '''Summarize this CXR report into a short, concise paragraph by focusing only on the key medical findings and omitting any references to prior events, personal details, or specific times.'''

if input('Do you want to use taobao key? (y/n)') == 'y':
    client = OpenAI(api_key=...)
else:
    client = OpenAI(api_key=...)
# List to store cleaned reports
cleaned_reports = []
associated_path = []
runned = int(input('Enter the number of reports already cleaned: '))
# Iterate through raw reports and generate cleaned reports
for count, (path, content) in enumerate(tqdm(zip(dicom_paths, reports), total=len(reports))):
    if count < runned:
        continue
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [{"role": "system", "content": SYS_MSG},
                {"role": "user", "content": content}],
            stream=False,
            temperature=1.0,
        )
        cleaned_reports.append(response.choices[0].message.content)
        associated_path.append(path)
    except Exception as e:
        print(e)
        pass
    if len(associated_path) == 10000:
        break
print('number of cleaned reports:', len(cleaned_reports))
print(len(cleaned_reports))
# Output cleaned reports as JSON
output_data = {
    "files": associated_path,
    "cleaned_reports": cleaned_reports,
}

# Save output as JSON
directory = 'openai_output'
filename = f'cleaned_reports_({runned}-{runned+10000}).json'
path = os.path.join(directory, filename)

# Check if the directory exists, if not create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Write data to a file with proper formatting
with open(path, 'w') as outfile:
    json.dump(output_data, outfile, indent=4)

print("Cleaning and saving reports completed.")
