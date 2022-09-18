from torchvision import datasets, transforms
from base import BaseDataLoader   
from typing import List, Dict
import random
import pandas as pd
import numpy as np
import torch

from base.base_dataset import BaseDataSet

class HymenopteraDataLoader(BaseDataLoader):
    
    """
    Hymenoptera data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.2, num_workers=1, training=True):
        
        trsfm = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CaesarEng():
          
    def __init__(self, key = None, symb_alphabet = None):
        if not key:
            key = random.randint(1, 100)     
        if not symb_alphabet:
            self.alphabet = [chr(i) for i in [10, *range(32,92), *range(92,127)]]
        else:
            self.alphabet = symb_alphabet
        self.__key = key % len(self)  
        self.encode_table = dict(zip(range(len(self.alphabet)),self.alphabet))
        
    def encode(self, symb_seq: str):
        key_seq = [self.get_key_by_value(self.encode_table, s) + 
                   self.__key for s in symb_seq]
        return "".join(self.encode_table[i] for i in self.clip_seq(key_seq))
        
    def decode(self, symb_seq: str):
        return "".join(el for el in [self.encode_table[self.clip_val(v-self.__key)] 
                                     for v in [self.get_key_by_value(self.encode_table, s) 
                                               for s in symb_seq]])
    
    def clip_seq(self, int_seq: List[int]):
        for i, v in enumerate(int_seq):
            int_seq[i] = self.clip_val(v)
        return int_seq 
    
    def clip_val(self, value: int):
        if value<0:
            return value+len(self)  
        elif value>=len(self):
            return value-len(self)  
        return value      
    
    def __len__(self):
        return len(self.alphabet)
    
    def get_key(self):
        print(self.__key)
        
    @staticmethod
    def get_key_by_value(d: Dict, v):
        assert len(set(d.values())) == len(d.values())
        assert v in d.values()
        return [key for key, value in d.items() if value==v][0]

class CaesarEngDataLoader(BaseDataLoader):
    """
    Class for data loading eng text with Caesar
    """
    def __init__(self, data_dir, batch_size, max_len=50, shuffle=False, 
                 validation_split=0.2, num_workers=6, collate_fn=False, 
                 training = True, key = None, step_for_samples = 5):
        with open(data_dir, encoding='utf-8') as txt_file:
            txt_file.seek(3)
            self.text = txt_file.read()
            txt_file.close()
        symbols = [chr(i) for i in [10, *range(32,92), *range(92,127)]]
        self.max_len = max_len
        self.caesar = CaesarEng(key, symbols)
        self.encoded_text = self.caesar.encode(self.text)
        self.index_to_char = sorted(list(set(self.text).union(set(self.encoded_text))))
        self.char_to_index = {c: i for i, c in enumerate(self.index_to_char)}
        print(self.char_to_index)
        self.dataset = torch.utils.data.TensorDataset(*self.create_samples(step_for_samples))
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
    def create_samples(self, step):
        X = torch.zeros(((len(self.text) - self.max_len)//step + 1, 
                         self.max_len), dtype=torch.int64)
        y = torch.zeros(((len(self.text) - self.max_len)//step + 1, 
                         self.max_len), dtype=torch.int64)
    
        for i in range(0, len(self.text) - self.max_len, step):
            X[i//step] = torch.tensor([self.char_to_index[char] 
                                       for char in self.encoded_text[i: i + self.max_len]])
            y[i//step] = torch.tensor([self.char_to_index[char] 
                                       for char in self.text[i: i + self.max_len]])
        
        return X, y
    
class SeqGen():
    def __init__(self, max_len) -> None:
        self.max_len = max_len
        
    def generate_x(self):
        self.x = torch.randint(10,(self.max_len,))
        return self.x
        
    def encode_x(self):
        self.y = torch.zeros((self.max_len,), dtype=int)
        for i, x in enumerate(self.x):
            if (i==0):
                self.y[i] = x
            elif (x + self.x[0] >= 10):
                self.y[i] = x + self.x[0] - 10
            else:
                self.y[i] = x + self.x[0]
        return self.y
    
    def generate_x_y(self):
        return self.generate_x(), self.encode_x()
    
class SeqDataSet(BaseDataSet):
    def __init__(self, max_len=50, total_size=100) -> None:
        self.total_size = total_size
        self.generator = SeqGen(max_len)
        
    def __getitem__(self, *inputs):        
        return self.generator.generate_x_y()
    
    def __len__(self):
        return self.total_size
    
class SeqDataLoader(BaseDataLoader):
    """
    Class for data loading seq of numbers
    """
    def __init__(self, batch_size, max_len=50, shuffle=False, 
                 validation_split=0.2, num_workers=6, collate_fn=False, 
                 training = True, total_size = 100):
        self.max_len = max_len
        self.dataset = SeqDataSet(max_len, total_size)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

