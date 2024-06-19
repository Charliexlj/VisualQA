import json

# Load the JSON file
with open('output/llama_8B_instruct/new_labels.json', 'r') as file:
    data = json.load(file)

y_labels = data['raw_label']
concise_labels = data['concise_label']

# Initialize variables to count matches and total comparisons
matches = 0
total = 0

# Compare the labels
for y, concise in zip(y_labels, concise_labels):
    matches += sum(1 for y_item, concise_item in zip(y, concise) if y_item == concise_item and y_item == 1)
    total += sum(1 for y_item, concise_item in zip(y, concise) if concise_item == 1 or y_item == 1)

# Calculate accuracy
accuracy = matches / total * 100  # convert to percentage

print(f'Accuracy of y_label matching concise_label: {accuracy:.2f}%')