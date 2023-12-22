from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.io import read_image
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class MyDatasets(Dataset):
    def __init__(self, directory = None, transform = None):
        
        self.directory = os.path.join(os.path.dirname(__file__), directory)
        self.transform = transform
        
        self.label, self.label2index = self.findClasses()
        self.img_labels = self.createImagesData()


    def findClasses(self):
        classes = [d.name for d in os.scandir(self.directory)]
        classes.sort()
        class2index = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class2index

    def createImagesData(self):
        if self.directory:
            img_labels = []
            for target_label in sorted(self.label2index):
                target_dir = os.path.join(self.directory, target_label)

                for root, _, file_names in sorted(os.walk(target_dir, followlinks = True)):
                    for file_name in file_names:
                        img_path = os.path.join(root, file_name)
                        
                        img = read_image(path=img_path)
                        if self.transform:
                            img = self.transform(img)
                            
                        img_and_label = img, target_label
                        img_labels.append(img_and_label)
                        
                    
            
        logger.debug(f"Label: {target_label} | Number of images: {len(img_labels)}")
        logger.debug(f'img_labels: {img_labels}')
        return img_labels
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img, label = self.img_labels[index]
        return img, label
    
if __name__ == "__main__":
    dataset = MyDatasets(directory = "images", transform = None)
    logger.debug(len(dataset))