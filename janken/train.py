import torch
from .datasets import MyDatasets

def pre_process():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    
    SEED = 1
    torch.manual_seed(SEED)
    
    return 
    

def main():
    pre_process()
    
    
if __name__ == "__main__":
    main()