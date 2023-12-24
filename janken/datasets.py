from cProfile import label
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.io import read_image
from PIL import Image
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class MyDatasets(Dataset):
    def __init__(self, directory = 'images', transform = None):
        
        self.directory = os.path.join(os.path.dirname(__file__), directory)
        self.transform = transform
        
        self.labels, self.target, self.label2index = self.findClasses()
        self.img_and_targets = self.createImagesData()


    def findClasses(self):
        label = [d.name for d in os.scandir(self.directory)]
        label.sort()
        target = F.one_hot(torch.arange(len(label)))
        label2index = {label[i]: i for i in range(len(label))}
        return label, target, label2index

    def createImagesData(self):
        img_and_targets = []
        if self.directory:
            for target_label in self.labels:
                target_dir = os.path.join(self.directory, target_label)
                label_index = self.label2index[target_label]
                _target = self.target[label_index]
                
                for file_name in os.listdir(target_dir):
                    img_path = os.path.join(target_dir, file_name)
                    
                    # img = read_image(path=img_path)
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                        
                    img_and_target = img, _target
                    img_and_targets.append(img_and_target)
                        
                logger.debug(f"Label: {target_label} id: {label_index} | Number of images: {len(img_and_targets)}")
                        
                    
            
        logger.debug(f'img_and_targets: {img_and_targets[0]}')
        return img_and_targets
    
    def __len__(self):
        return len(self.img_and_targets)

    def __getitem__(self, index):
        img, label = self.img_and_targets[index]
        return img, label
    
if __name__ == "__main__":
    dataset = MyDatasets(directory = "images", transform = None)
    logger.debug(len(dataset))