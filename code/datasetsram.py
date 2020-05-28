import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def dummytrsfrm(data):
    return data

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', partnum = 3, shuffle = False):
        if transforms_ is not None:
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = dummytrsfrm
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        
        self.minlen = min(len(self.files_A), len(self.files_B))
        
        if len(self.files_B) == self.minlen:
            self.files_B.extend(self.files_B[:len(self.files_A) - self.minlen])
        else:
            self.files_A.extend(self.files_A[:len(self.files_B) - self.minlen])
        
        self.minlen = min(len(self.files_A), len(self.files_B))
        
        self.part = 1
        self.data_A = []
        for i in range(self.minlen//3):
            im = Image.open(self.files_A[i])
            imcopy = im.copy()
            self.data_A.append(self.transform(imcopy))
            im.close()
            
        self.data_B = []
        for i in range(self.minlen//3):
            im = Image.open(self.files_B[i])
            imcopy = im.copy()
            self.data_B.append(self.transform(imcopy))
            im.close()
        
        if shuffle:
            random.shuffle(self.data_A)
            random.shuffle(self.data_B)

    def __getitem__(self, index):
        if index >= self.minlen*self.part:
        
            start = self.minlen*part//3
            
            if part == 2:
                end = self.minlen
            else:
                part += 1
                end = self.minlen*part
            
            self.data_A = []
            for i in range(start,end):
                im = Image.open(self.files_A[i])
                imcopy = im.copy()
                self.data_A.append(self.transform(imcopy))
                im.close()
            
            self.data_B = []
            for i in range(start,end):
                im = Image.open(self.files_B[i])
                imcopy = im.copy()
                self.data_B.append(self.transform(imcopy))
                im.close()
            
            if shuffle:
                random.shuffle(self.data_A)
                random.shuffle(self.data_B)
            
        item_A = self.data_A[index]

        if self.unaligned:
            item_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
        else:
            item_B = self.data_B[index]

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
