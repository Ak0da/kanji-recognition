import torch

import random
import math
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from torchinfo import summary
from torchvision import transforms

import pathlib
from os import listdir
from os.path import isfile, join

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
class kanji_detector(torch.nn.Module):
    def __init__(self,nb_symbols=2199,dropout1=0.4, dropout2=0.2, dropout3=0.2):
        super(kanji_detector, self).__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,17),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,64,5),
            torch.nn.Dropout(dropout2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64,128,3),
            torch.nn.Dropout(dropout3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(2048,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,nb_symbols)
            #torch.nn.Softmax(-1)
        )
        
    def forward(self, input):
        out = self.sequence(input)
        return out
"""

"""
class kanji_detector(torch.nn.Module):
    def __init__(self,nb_symbols=2199,dropout1=0.4, dropout2=0.2, dropout3=0.2):
        super(kanji_detector, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,32,17)
        self.act1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(dropout1)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32,64,5)
        self.act2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(dropout2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(64,128,3)
        self.act3 = torch.nn.ReLU()
        self.drop3 = torch.nn.Dropout(dropout3)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.flat = torch.nn.Flatten()
        self.lin1 = torch.nn.Linear(2048,512)
        self.act4 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(512,nb_symbols)
        
        
    def forward(self, input):
        out = self.conv1(input)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.act2(out)
        out = self.drop2(out)
        out = self.pool2(out)
        
        out = self.conv3(out)
        out = self.act3(out)
        out = self.drop3(out)
        out = self.pool3(out)
        
        out = self.flat(out)
        out = self.lin1(out)
        out = self.act4(out)
        out = self.lin2(out)
        
        return out
"""
"""
def kanji_detector(nb_symbols=2199,dropout1=0.4, dropout2=0.2, dropout3=0.2):
    return torch.nn.Sequential(
            torch.nn.Conv2d(1,32,17),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,64,5),
            torch.nn.Dropout(dropout2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64,128,3),
            torch.nn.Dropout(dropout3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(2048,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,nb_symbols)
            #torch.nn.Softmax(-1)
        )
"""
        
def kanji_detector(nb_symbols=2199,dropout1=0.4, dropout2=0.2, dropout3=0.2):
    return torch.nn.Sequential(
            torch.nn.Conv2d(1,32,17),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,64,5),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64,128,3),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout3),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(2048,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,nb_symbols)
            #torch.nn.Softmax(-1)
        )



