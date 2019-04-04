import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import PIL
import numpy as np
import imageio
from glob import glob

numbers = (
        '<_>',
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '.',
        # '/',
        # '@',
        # '&',
        # ',',
        # '(',
        # ')',
        # '*',
        # "'",
        # '-',
        # ':',
        # '%',
        # '=',
        # '#',
        # '$',
        # '!',
        # '?',
        # '~',
        # '_',
        # '+',
        # '"',
        'A',
        # '\/',
        # '\\',
        # ';',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I',
        'J',
        'K',
        'L',
        'M',
        'N',
        'O',
        'P',
        'Q',
        'R',
        'S',
        'T',
        'U',
        'V',
        'W',
        'X',
        'Y',
        'Z',
        # '[',
        # ']',
        # '|',
        # '<',
        # '>',
        # '^',
        # '{',
        # '}',
        # '·'
        )
    
class DashboardDataset(Dataset):
    
    def __init__(self, path, transform):
        self.transform = transform
        self.class_to_ind = dict(zip(numbers, range(len(numbers))))
        self.ind_to_class = dict(zip(range(len(numbers)), numbers))
        self.num_of_classes = len(numbers)

        with open(path + '/images.pkl', 'rb') as handle:
            self.data = pickle.load(handle)
        with open(path + '/labels.pkl', 'rb') as handle:
            self.targets = pickle.load(handle)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.targets[idx]
        length = len(target)
        #print(np.any(np.isnan(self.data[idx]))
        try:
            image = PIL.Image.fromarray(self.data[idx])
        except:
            print(self.data[idx].shape)
        return self.transform(image), target, length

class OCRDataset(Dataset):

    def __init__(self, path, transform, is_train, length=[5, 8, 10, 15]):
        folder = 'images' if is_train else 'images_val'
        self.paths = []
        for l in length:
            self.paths += sorted(glob(path + '/{}/{}/odo*.jpg'.format(l, folder)))
        self.paths = [i for i in self.paths if '_' in i]
        self.transform = transform
        self.class_to_ind = dict(zip(numbers, range(len(numbers))))
        self.ind_to_class = dict(zip(range(len(numbers)), numbers))
        self.num_of_classes = len(numbers)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx].replace('images', 'labels').replace('jpg', 'pkl'), 'rb') as f:
            target = pickle.load(f)
        length = len(target)
        try:
            image = PIL.Image.fromarray(imageio.imread(self.paths[idx]))
        except:
            print(self.paths[idx])
        if len(image.size) == 3:
            print(len(image.size) == 3)

        return self.transform(image), target, length


def build_data_loader(path, batch_size, num_workers, transforms, is_train):
    return DataLoader(
        OCRDataset(path, transforms, is_train),
        batch_size,
        num_workers = num_workers,
        collate_fn = collate_fn,
        shuffle=True if is_train else False
    )

def collate_fn(batch):
    data = []
    target = []
    lengths = []
    for i in batch:
        data.append(i[0])
        target.append(torch.tensor(i[1]))
        lengths.append(i[2])
    return torch.stack(data, dim=0), target, torch.tensor(lengths)