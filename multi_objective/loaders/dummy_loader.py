import torch
from torch.utils import data

class DUMMY(data.Dataset):
    
    def __init__(self, split='train',**kwargs):
        assert split in ["train", "val", "test"]
        
    def __len__(self):
        """__len__"""
        return 1
    
    def __getitem__(self, index):
        return dict()
    
    def task_names(self):
        return None
    
if __name__ == '__main__':
    d = DUMMY()
    loader = data.DataLoader(d, batch_size=5, shuffle=True)
    for batch in loader:
        print(batch)