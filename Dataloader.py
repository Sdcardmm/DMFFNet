import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import MyDataset

file = pd.read_excel(r'data/gene/result.xlsx', dtype={'住院号': str})
folder_path = r'data/gene/tumor'

file_names = [name.split('.')[0] for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]

file = file[file['住院号'].isin(file_names)].reset_index(drop=True)

if file.empty:
    raise ValueError("After filtering, the DataFrame is empty.")

file['patient_id'] = pd.Categorical(file['住院号'], categories=file_names, ordered=True)
file = file.sort_values(by='住院号').reset_index(drop=True)

file_names = file['住院号'].tolist()

image_paths = [os.path.join(folder_path, f'{file_name}.png') for file_name in file_names]
label_list = file['腹膜'].tolist()

if not image_paths or not label_list:
    raise ValueError("Image paths or labels list is empty.")

for file_name, path in zip(file_names, image_paths):
    if not os.path.exists(path):
        raise ValueError(f"File does not exist: {path}")

dataset = MyDataset(image_paths=image_paths, label_paths=label_list,
                    transform=transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5])
                    ]))

if len(dataset) == 0:
    raise ValueError("Dataset is empty.")

if __name__ == '__main__':
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(train_loader.dataset.__getitem__(0)[0].shape)