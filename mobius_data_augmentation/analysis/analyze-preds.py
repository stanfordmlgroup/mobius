import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_excel('predictions-with-mobius.xls')
#df = pd.read_excel('predictions-without-mobius.xls')
gt_label_col = 'Actual_number (0 index)'
pred_label_col = 'Predicted_number (0 index)'

correctness_col = 'correctness'
df[correctness_col] = df[gt_label_col] == df[pred_label_col]

print(df.head())

# These classes in the mapping are from the column Actual labels category
gt_label_name_col = 'Actual labels category'
print(f'All superset categories: {df[gt_label_name_col].unique()}')
class2biomap =   { 
                    'aquatic mammals': True,
                    'fish': True,
                    'flowers': True, # 
                    'food containers': False, # 
                    'fruit and vegetables': True, # 
                    'household electrical device': False,
                    'household furniture': False,                    
                    'insects': True,
                    'large carnivores': True,
                    'large man-made outdoor things': False,
                    'large natural outdoor scenes': False, # 
                    'large omnivores and herbivores': True,
                    'non-insect invertebrates': True,
                    'medium-sized mammals': True,
                    'people': True,
                    'reptiles': True,
                    'small mammals': True,
                    'trees': True, # 
                    'vehicles 1': False,
                    'vehicles 2': False,
                 }

correctness_dict = defaultdict(list)
class_correctness_dict = defaultdict(list)
for i, row in df.iterrows():
    cl = row[gt_label_name_col]
    corr = row[correctness_col]
    if class2biomap[cl]:
        # Biological
        correctness_dict['bio'].append(corr)
    else:
        # Non-biological
        correctness_dict['nonbio'].append(corr)
    class_correctness_dict[cl].append(corr)

for b, corr_list in correctness_dict.items():
    print(f'{b}: {len(corr_list)} examples, with mean {np.mean(corr_list)} std dev {np.std(corr_list)}')

for cls, corr_list in class_correctness_dict.items():
    print(f'{cls}: {len(corr_list)} examples, with mean {np.mean(corr_list)} std dev {np.std(corr_list)}')

