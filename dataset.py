import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from skimage import io 
from PIL import Image
import config as cf

class FaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.images = pd.read_csv(csv_file)
        self.images = self.images[['path', 'label']]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.images.iloc[idx, 0])
        #image = io.imread(img_name)
        image = Image.open(img_name)
        label = self.images.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    csv_file = '/home/nhatkhang/Documents/GANs/FakeReal/archive/test.csv'
    root_dir = '/home/nhatkhang/Documents/GANs/FakeReal/archive/real_vs_fake/real-vs-fake'
    faces = FaceDataset(csv_file, root_dir)

    data = DataLoader(faces, batch_size=4, shuffle=True)
    
    images, label = next(iter(data))

    print(images.shape)

    shape = images.shape
    images = images.reshape(shape[0],shape[3],shape[1], shape[2])
    print(images.shape)

