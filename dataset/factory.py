from dataset import dataset
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_dataset(datadir, split, model_name, transform_name):
    study_dataset = dataset.StudyDataset(
        datadir=datadir,
        split=split,
        model_name=model_name,
        transform_name=transform_name,
    )
    return study_dataset

def creat_dataloader(dataset, batch_size, shuffle):
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        )
    return dataloader

def train_val_split(base_path, ratio):
    train_data_path = os.path.join(base_path, "train_data.npy")
    train_target_path = os.path.join(base_path, "train_target.npy")
    val_data_path = os.path.join(base_path, "val_data.npy")
    val_target_path = os.path.join(base_path, "val_target.npy")

    train_data = np.load(train_data_path)
    train_target = np.load(train_target_path)
    train_data, val_data, train_target, val_target = train_test_split(train_data, train_target,
                                                                        test_size=ratio,
                                                                        stratify=train_target)
    np.save(train_data_path, train_data)
    np.save(train_target_path, train_target)
    np.save(val_data_path, val_data)
    np.save(val_target_path, val_target) 

class SoftCrop: 
    '''
    crop image
    
    '''
    def __init__(self, n_class=10,
                 sigma_crop=0.3, k=2,  bg_intensity=1.0):
        
        self.n_class = n_class

        # crop parameters
        self.k = k
        self.sigma_crop = sigma_crop
        self.bg_intensity = bg_intensity


    def draw_offset(self, sigma, limit, n=100):
        # draw an integer from gaussian within +/- limit
        for _ in range(n):
            x = torch.randn((1)) * (sigma * limit)
            if abs(x) <= limit:
                return int(x)
        return int(0)
    
    def compute_conf(self, overlap, k):
        p_min = torch.tensor(1 / self.n_class, dtype=torch.float32)
        visibility = torch.tensor(overlap, dtype=torch.float32)
        confidence = 1 - (1 - p_min) * (1 - visibility) ** k
        confidence = torch.max(confidence, p_min)
        return confidence
    
    def softing_targets(self, label, confidence):
        remain_prob = (1 - confidence) / (self.n_class - 1)
        adjusted_prob = torch.full((self.n_class,), remain_prob) 
        adjusted_prob[label] = confidence
        return adjusted_prob
    
    def __call__(self, image, label):

        dim1 = image.size(1)
        dim2 = image.size(2)

        # create a 3x by 3x sized noise background
        bg = torch.ones((3, dim1*3, dim2*3)) * self.bg_intensity *  torch.randn((3, 1, 1))
        bg[:, dim1:2*dim1, dim2:2*dim2] = image     # put image at the center patch
        offset1 = self.draw_offset(self.sigma_crop, dim1)
        offset2 = self.draw_offset(self.sigma_crop, dim2)
        
        # calculate offset/delta
        left = offset1 + dim1
        top = offset2 + dim2
        right = offset1 + dim1 * 2
        bottom = offset2 + dim2 * 2

        # compute confidence of the label of the train data
        intersection = (dim1 - abs(offset1)) * (dim2 - abs(offset2))
        overlap = intersection / (dim1 * dim2)
        confidence = self.compute_conf(overlap, self.k)
        if confidence.item() > 1 or confidence.item() < 0:
            raise ValueError(f"Confidence should be higher than 0 or lower than 1. Now {confidence}")

        softened_target = self.softing_targets(label, confidence)
        new_image = bg[:, left: right, top: bottom]

        return new_image, softened_target
