import json

with open('output/eval/bleu_rouge_meteor.json') as f:
    data = json.load(f)
    
avg_bleu_1 = sum([item['BLEU-1'] for item in data])/len(data)
avg_bleu_2 = sum([item['BLEU-2'] for item in data])/len(data)
avg_bleu_3 = sum([item['BLEU-3'] for item in data])/len(data)
avg_bleu_4 = sum([item['BLEU-4'] for item in data])/len(data)
avg_rouge_1 = sum([item['ROUGE-1'] for item in data])/len(data)
avg_METEOR = sum([item['METEOR'] for item in data])/len(data)

print(f'Average BLEU-1: {avg_bleu_1}')
print(f'Average BLEU-2: {avg_bleu_2}')
print(f'Average BLEU-3: {avg_bleu_3}')
print(f'Average BLEU-4: {avg_bleu_4}')
print(f'Average ROUGE-1: {avg_rouge_1}')
print(f'Average METEOR: {avg_METEOR}')