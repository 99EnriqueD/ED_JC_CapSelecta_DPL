import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
# You might want to change path to the corresponding one
# IMPORTANT: offset the file right!!
path_train_index ="data/anno_fine/train.txt"
path_train_label ="data/anno_fine/train_cate.txt"
path_test_index = "data/anno_fine/test.txt"
path_test_label = "data/anno_fine/test_cate.txt"
path_root_dir_images= "data/images/"

class FashionTrainDataset(Dataset):

    def __init__(self, csv_file = path_train_index, label_file= path_train_label, root_dir = path_root_dir_images, transform=None):
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
        if self.transform:
            image = self.transform(image)
        return image,landmarks

class FashionTestDataset(Dataset):

    def __init__(self, csv_file = path_test_index, label_file= path_test_label, root_dir = path_root_dir_images, transform=None):
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
        if self.transform:
            image = self.transform(image)
        return image,landmarks


#transformz = transforms.Compose([
 #       transforms.ToTensor(),
  #      transforms.RandomResizedCrop(224),
   #     transforms.Normalize((0.5), (0.5))
    #])
#data=FashionTrainDataset(transform= transformz)
#print(data[0][1])