from sympy import E
import torch
from torch.utils.data import DataLoader, random_split
from datasets import MyDatasets
from model import Model
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def pre_process():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    
    SEED = 1
    torch.manual_seed(SEED)
    
    SPRIT_RATIO = 0.8
    dataset = MyDatasets(directory = "images", transform = None)
    return torch.utils.data.random_split(dataset, [int(len(dataset) * SPRIT_RATIO), len(dataset) - int(len(dataset) * SPRIT_RATIO)], generator = torch.Generator().manual_seed(SEED))
    

def main():
    train_dataloader, val_dataloader = pre_process()
    
    
        
    
    
    
    
    
    
if __name__ == "__main__":
    main()