from abc import abstractmethod
import torch

class BaseDataSet(torch.utils.data.Dataset):
    
    @abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, *inputs):
        raise NotImplementedError
    

