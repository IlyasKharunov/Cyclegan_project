import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, aligned=True, mode='train', place = 'drive'):
        self.transform = transforms.Compose(transforms_)
        self.aligned = aligned
        self.place = place
        
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        
        if place == 'ram':
            self.files_A = [self.transform(Image.open(self.files_A[index])) for index in range(len(self.files_A))]
            self.files_B = [self.transform(Image.open(self.files_B[index])) for index in range(len(self.files_B))]
    def __getitem__(self, index):
        id_a = index % len(self.files_A)
        
        if self.aligned:
            id_b = index % len(self.files_B)
        else:
            id_b = random.randint(0, len(self.files_B) - 1)
            
        if self.place == 'ram':
            item_A = self.files_A[id_a]
            item_B = self.files_B[id_b]
        else:
            item_A = self.transform(Image.open(self.files_A[id_a]))
            item_B = self.transform(Image.open(self.files_B[id_b]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))