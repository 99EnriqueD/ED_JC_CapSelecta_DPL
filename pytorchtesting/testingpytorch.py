import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform

path_lijst ="pytorchtesting/lijst.txt"
path_label ="pytorchtesting/label.txt"

class FashionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file = path_lijst, label_file= path_label, root_dir = 'pytorchtesting/', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        
        self.label_frame= pd.read_csv(label_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.label_frame.iloc[idx, 0]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        return image,landmarks

        if self.transform:
            sample = self.transform(sample)

        return sample
