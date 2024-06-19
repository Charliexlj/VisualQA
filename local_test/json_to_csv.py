import json
import pandas as pd
import pandas as pd

with open('output/xraygpt/impressions.json', 'r') as file:
    json_data = json.load(file)

csv_data = pd.read_csv('output/test_list.csv')
    
image_idx = csv_data['Image Index'].tolist()
report = json_data['impressions_pred']

print(len(image_idx))
print(len(report))

df = pd.DataFrame({'Image Index': image_idx, 'Report': report})
df.to_csv('output/xraygpt/xraygpt_reports.csv', index=False)
