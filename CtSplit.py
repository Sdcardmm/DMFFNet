import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

excel_file = '../data/医大三胃癌.xlsx'
image_folder = 'F:\\workspaces\\workspace-pycharm\\first\\data\\fat_3'
output_folder = 'F:\\workspaces\\workspace-pycharm\\first\\data\\fat_3_in'
df = pd.read_excel(excel_file)

ids = df['住院号'].astype(str).str.zfill(6)
labels = df['腹膜']

ids_0 = ids[labels == 0]
ids_1 = ids[labels == 1]

train_ids_0, temp_ids_0 = train_test_split(ids_0, test_size=0.4, random_state=42)
val_ids_0, test_ids_0 = train_test_split(temp_ids_0, test_size=0.5, random_state=42)

train_ids_1, temp_ids_1 = train_test_split(ids_1, test_size=0.4, random_state=42)
val_ids_1, test_ids_1 = train_test_split(temp_ids_1, test_size=0.5, random_state=42)

train_ids = pd.concat([train_ids_0, train_ids_1])
val_ids = pd.concat([val_ids_0, val_ids_1])
test_ids = pd.concat([test_ids_0, test_ids_1])

train_folder = f'../{output_folder}/12345'
val_folder = f'../{output_folder}/00000'
test_folder = f'../{output_folder}/67890'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for id in train_ids:
    image_path = os.path.join(image_folder, f'{id}.png')
    if os.path.exists(image_path):
        shutil.copy(image_path, train_folder)

for id in val_ids:
    image_path = os.path.join(image_folder, f'{id}.png')
    if os.path.exists(image_path):
        shutil.copy(image_path, val_folder)

for id in test_ids:
    image_path = os.path.join(image_folder, f'{id}.png')
    if os.path.exists(image_path):
        shutil.copy(image_path, test_folder)

print("数据集划分完成，图像文件已移动到相应的文件夹。")