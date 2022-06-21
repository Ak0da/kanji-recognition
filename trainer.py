

import torch
import hub
import random
import math
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from torchinfo import summary
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd


import copy
import time
import json
import pathlib
import os
from os import listdir
from os.path import isfile, join

device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 64
nb_symbols = 2199

from kanji_detection_model import kanji_detector

    
def getModel():
    return kanji_detector()


def testModel():
    modelRunnable = getModel().to(device=device)
    print(modelRunnable)
    
    summary1 = summary(
        modelRunnable,
        input_size=[
            (20, 1, image_size, image_size)
        ],
        dtypes=[torch.double, torch.double],
        depth=3
    )
    
    print(summary1)
    
    del modelRunnable
    torch.cuda.empty_cache()


#print("Loading training files")
trainingPath = pathlib.Path().resolve() / "Training_set"

"""
picturesNames = [f for f in listdir(trainingPath) if isfile(join(trainingPath, f))]

g_dictNames = {name:{'name':name , 'number':int(name.split('_')[0]) , 'symbol':name.split('_')[1]} for name in picturesNames}
#g_tempDictNames = {name:{'name':name, 'number':int(name.split('_')[0]) , 'symbol':name.split('_')[1]} for name in picturesNames}

shufflePicturesNames = picturesNames.copy()
random.shuffle(shufflePicturesNames)
"""

#------------------------------------------------------------------------------------------------


class KanjiImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, delimiter=',', header=0)
        #print(self.img_labels)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
img_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(image_size),
    transforms.ToTensor()
])
training_data = KanjiImageDataset("loader_data.csv", ".\Training_set", img_transform, int)


selectorIndex = 0
def selectBatches(batch_size) -> list:
    global selectorIndex
    global shufflePicturesNames
    
    true_list_len = len(shufflePicturesNames)
    batch = []
    while len(batch) < batch_size:
        list_len = true_list_len - selectorIndex
        n_to_find = batch_size - len(batch)
        
        if list_len >= n_to_find:
            for i in range (n_to_find):
                name = shufflePicturesNames[selectorIndex]
                item = g_dictNames[name]
                batch.append(item)
                selectorIndex += 1
                
        else:
            batch.extend([g_dictNames[name] for name in shufflePicturesNames[selectorIndex:]])
            random.shuffle(shufflePicturesNames)
            selectorIndex = 0
            
            
    
    #print(batch[0])
    #print(batch_size)
    #print(len(batch))
    
    return batch

def getAnswerIndices(batchList) -> torch.FloatTensor:
    
    correctAnswer = torch.zeros((len(batchList),nb_symbols)).float()
    correctAnswerIndices = torch.zeros(len(batchList)).long()
    
    for i in range(len(batchList)):
        #indexCorrect = batchList[i][1]['number']-1
        indexCorrect = batchList[i]['number']-1
        correctAnswer[i][indexCorrect] = 1
        correctAnswerIndices[i] = indexCorrect
    
    
    return correctAnswer, correctAnswerIndices


#def countCorrect(answer: torch.FloatTensor, correctAnswer: torch.FloatTensor):
def countCorrect(answer: torch.FloatTensor, correctAnswerIndices: torch.FloatTensor):
    
    _,indicesAnswer = torch.max(answer, dim=1)
    
    #print(indicesAnswer)
    #print(indicesCorrect)
    numCorrect = (indicesAnswer == correctAnswerIndices).long().sum()
    
    return numCorrect.item()

#def countTop5Correct(answer: torch.FloatTensor, correctAnswer: torch.FloatTensor):
def countTop5Correct(answer: torch.FloatTensor, correctAnswerIndices: torch.FloatTensor):
    _,indicesAnswer = answer.topk(k=5, dim=1)
    
    numCorrect = (indicesAnswer == correctAnswerIndices).long().sum()
    
    return numCorrect.item()
    
def getPictures(batchList) -> torch.FloatTensor:
    
    images = torch.zeros((len(batchList), 3, 40, 40)).to(device=device)
    to_tensor = transforms.ToTensor()
    to_grey = transforms.Grayscale()
    
    convert_tensor = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(image_size)
        #transforms.ToTensor()
    ])
    
    
    for i in range(len(batchList)):
        item = batchList[i]
        #filename = item[0]
        filename = item['name']
        img = Image.open(trainingPath / filename)
        #images[i,:,:,:] = convert_tensor(img)
        images[i,:,:,:] = to_tensor(img).to(device=device)
    
    images = convert_tensor(images)
        
    return images


def train(model, n_epoch, batch_size, lr):
    n_batches = 100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #dataset = datasets.ImageFolder(trainingPath, transform=transform)
    dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model.train()
    loss_f = torch.nn.CrossEntropyLoss()
    best_percent = 0
    for epoch in range(n_epoch):
        
        #print("Epoch " + str(epoch+1) + " is running")
        n_correct_1 = 0
        n_correct_5 = 0
        n_total = n_batches*batch_size
        t_loss = 0
        #for g in optimizer.param_groups:
            #g['lr'] = lr/(epoch+1)**0.5
        #model.zero_grad()
            
        time_select = 0 #debug
        time_pictures = 0 #debug
        time_model = 0 #debug
            
        
        for i in range(n_batches):

            model.zero_grad()
            #optimizer.zero_grad() not needed ?
            
            start = time.time() #debug
            #batch = selectBatches(g_dictNames, g_tempDictNames, batch_size)
            #batch = selectBatches(batch_size)
            images, correct_answer_indices = next(iter(dataloader))
            end = time.time() #debug
            time_select += end-start # debug
            
            start = time.time() #debug
            #images = getPictures(batch)
            end = time.time() #debug
            time_pictures += end-start # debug
            
            #correct_answer, correct_answer_indices = getAnswerIndices(batch)
            
            start = time.time() #debug
            answer = model(images.to(device=device))
            end = time.time() #debug
            time_model += end-start # debug
            
            loss = loss_f(answer,correct_answer_indices.to(device=device)).cpu()
            t_loss += loss.item()
            
            loss.backward()
            optimizer.step() #Trying at the end of the epoch ?
            
            if (epoch+1)%5 == 0:
                #n_correct_1 += countCorrect(answer, correct_answer.to(device=device))
                #n_correct_5 += countTop5Correct(answer, correct_answer.to(device=device))
                n_correct_1 += countCorrect(answer, correct_answer_indices.to(device=device))
                n_correct_5 += countTop5Correct(answer, correct_answer_indices.to(device=device))
            
            
            
            #print(loss.item())
            #print(torch.softmax(answer,dim=1))
            #print(correct_answer)
            #print(n_correct)
        
        #optimizer.step()
        adjust = 100
        percent_1 = math.floor(adjust*100*n_correct_1/n_total)/adjust
        percent_5 = math.floor(adjust*100*n_correct_5/n_total)/adjust
        display_loss = math.floor(adjust*t_loss)/adjust
        
        best_percent = percent_1 if percent_1 > best_percent else best_percent
        
        print("Time select : " + str(math.floor(time_select))) #debug
        print("Time pictures : " + str(math.floor(time_pictures))) #debug
        print("Time model : " + str(math.floor(time_model))) #debug
        
        print("Epoch " + str(epoch+1) +" loss : " + str(display_loss))
        
        if (epoch+1)%5 == 0:
            print("Epoch " + str(epoch+1) +" top-1 training accuracy : " + str(percent_1) + "%")
            print("Epoch " + str(epoch+1) +" top-5 training accuracy : " + str(percent_5) + "%")
        
        if percent_5 > 98.0:
            break
        
        print("")
        
    return best_percent


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.normal_(0, 0.001)
    
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.normal_(0, 0.001)


if __name__ == "__main__":        

    trainModel = getModel().to(device=device)
    weights_init(trainModel)
    n_epochs = 3 # 700

    print("Running on " + device + "\n")
    #train(trainModel, n_epochs, batch_sizes[0], learning_rates[0])
    train(trainModel, n_epochs, 100, 0.0003) #was 0.0001