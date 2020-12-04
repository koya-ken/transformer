import torch
from torch.utils.data import Dataset


class SinDataset(Dataset):
    def __init__(self,input_len:int=30,output_len:int=30,infer_type:str='future',startdeg:int=0,enddeg:int=360*3,length:int=1000,augumentation=False):
        self.indicies = torch.randint(startdeg, enddeg, length)
        self.input_len = input_len
        self.output_len = output_len
        self.infer_type = infer_type
        self.augumentation = augumentation
    
    def create_input(self, index):
        start_deg = self.indicies[index]
        degrees = torch.arange(start_deg,start_deg+self.input_len)
        rads = torch.deg2rad(degrees)
        return self.F(rads)

    def create_output(self, index):
        if self.infer_type == 'future':
            return self.create_output_future(index)
        elif self.infer_type == 'past':
            return self.create_output_past(index)
        elif self.infer_type == 'same':
            return self.create_input(index)

    def create_output_future(self, index):
        start_deg = self.indicies[index]
        start_deg = start_deg + self.input_len
        degrees = torch.arange(start_deg,start_deg+self.output_len)
        rads = torch.deg2rad(degrees)
        return self.F(rads)

    def create_output_past(self, index):
        start_deg = self.indicies[index]
        start_deg = start_deg - self.output_len
        degrees = torch.arange(start_deg,start_deg+self.output_len)
        rads = torch.deg2rad(degrees)
        return self.F(rads)
    
    def F(self, X):
        if self.augumentation:
            X = X + torch.randn_like(X) * 0.01
        return torch.sin(X) * torch.sin(torch.cos(X))

    def __len__(self):
        return len(self.indicies)
    
    def __getitem__(self, index):
        return self.create_input(index), self.create_output(index)