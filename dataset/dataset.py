import os, subprocess
import numpy as np
from torch.utils.data import Dataset
import utils
def download_dataset(url, output_dir):
    command = [
        'gdown',
        '--folder',
        url,
        '-O',
        output_dir,
    ]

    subprocess.run(command, check=True)

class StudyDataset(Dataset):
    def __init__(self, datadir, split, model_name, transform_name=None):
        # check for the supported split
        valid_split = ['train', 'val', 'test']
        if split not in valid_split:
            raise ValueError(f"Invalid split '{split}'. Choose one from {valid_split}")
        
        # check for the supported models
        valid_models = ['resnet50', 'resnet50_pretrained', 'vit', 'vit42', 'vit_pretrained']
        if model_name not in valid_models:
            raise ValueError(f"Invalid model {model_name}. Choose one from {valid_models}")

        # data load
        data_path = os.path.join(datadir, f"{split}_data.npy")
        target_path = os.path.join(datadir, f"{split}_target.npy")  
        self.data = np.load(data_path)
        self.target = np.load(target_path)

        # choose the transform
        self.transform_name = transform_name
        if transform_name == 'softcrop':
            self.transforms, self.custom_transforms = utils.choose_transform(model_name, transform_name)
        else:
            self.transforms = utils.choose_transform(model_name, transform_name)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        data = self.transforms(data)

        if self.transform_name == 'softcrop':
            data, target = self.custom_transforms(data, target)

        return data, target
