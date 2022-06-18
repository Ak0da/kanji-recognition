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




image_size = 64
nb_symbols = 2199
    
trainingPath = pathlib.Path().resolve() / "Training_set"
picturesNames = [f for f in listdir(trainingPath) if isfile(join(trainingPath, f))]

dictNames = {name:{'number':int(name.split('_')[0]) , 'symbol':name.split('_')[1]} for name in picturesNames}
tempDictNames = dictNames.copy()

#print(dictNames[picturesNames[0]])

def selectBatches(dictNames, tempDictNames, batch_size) -> list:
    batch=[]
    while len(batch) < batch_size:
        n_to_find = batch_size-len(batch)
        if len(tempDictNames) >= n_to_find:
            sample = random.sample(list(tempDictNames.items()), n_to_find)
            batch.extend(sample)
            for item in sample:
                del tempDictNames[item[0]]
        else:
            batch.extend(tempDictNames.items())
            tempDictNames = dictNames.copy()

    return batch

def getAnswerIndices(batchList) -> torch.FloatTensor:
    
    correctAnswer = torch.zeros((len(batchList),nb_symbols)).float()
    correctAnswerIndices = torch.zeros(len(batchList)).long()
    
    for i in range(len(batchList)):
        indexCorrect = batchList[i][1]['number']-1
        correctAnswer[i][indexCorrect] = 1
        correctAnswerIndices[i] = indexCorrect
    
    
    return correctAnswer, correctAnswerIndices

def countCorrect(answer: torch.FloatTensor, correctAnswer: torch.FloatTensor):
    
    
    _,indicesAnswer = torch.max(answer, dim=1)
    _,indicesCorrect = torch.max(correctAnswer, dim=1)
    
    #print(indicesAnswer)
    #print(indicesCorrect)
    numCorrect = (indicesAnswer == indicesCorrect).long().sum()
    
    return numCorrect.item()
    
def getPictures(batchList) -> torch.FloatTensor:
    
    images = torch.zeros((len(batchList), 1, image_size, image_size)).float()
    convert_tensor = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    
    for i in range(len(batchList)):
        item = batchList[i]
        filename = item[0]
        img = Image.open(trainingPath / filename)
        images[i,:,:,:] = convert_tensor(img)
        
    return images

def train(model, n_epoch, batch_size, lr):
    n_batches = 100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #dataset = datasets.ImageFolder(trainingPath, transform=transform)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    loss_f = torch.nn.CrossEntropyLoss()
    best_percent = 0
    for epoch in range(n_epoch):
        #print("Epoch " + str(epoch+1) + " is running")
        n_correct = 0
        n_total = n_batches*batch_size
        
        for g in optimizer.param_groups:
            g['lr'] = lr/(epoch+1)**0.5
        
        for i in range(n_batches):

            model.zero_grad()
            #optimizer.zero_grad()
            batch = selectBatches(dictNames, tempDictNames, batch_size)
            images = getPictures(batch)
            correct_answer, correct_answer_indices = getAnswerIndices(batch)

            answer = model(images.to(device=device))
            loss = loss_f(answer,correct_answer_indices.to(device=device)).cpu()
            
            loss.backward()
            optimizer.step()
            
            n_correct += countCorrect(answer, correct_answer.to(device=device))
            
            #print(loss.item())
            #print(torch.softmax(answer,dim=1))
            #print(correct_answer)
            #print(n_correct)

        adjust = 10000
        percent = math.floor(adjust*100*n_correct/n_total)/adjust
        best_percent = percent if percent > best_percent else best_percent
        print("Epoch " + str(epoch+1) +" training accuracy : " + str(percent) + "%\n")
        
    return best_percent


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.normal_(0, 0.001)
    
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.normal_(0, 0.001)

        
def eval(model, n_epoch, batch_size):
    n_batches = 1
    
    with torch.no_grad():

        model.eval()
        loss_f = torch.nn.CrossEntropyLoss()
        best_percent = 0
        for epoch in range(n_epoch):
            #print("Epoch " + str(epoch+1) + " is running")
            n_correct = 0
            n_total = n_batches*batch_size
            for i in range(n_batches):

                batch = selectBatches(dictNames, tempDictNames, batch_size)
                images = getPictures(batch)
                correct_answer, correct_answer_indices = getAnswerIndices(batch)

                answer = model(images.to(device=device))
                loss = loss_f(answer,correct_answer_indices.to(device=device)).cpu()

                print(answer)
                n_correct += countCorrect(answer, correct_answer.to(device=device))

            adjust = 10000
            percent = math.floor(adjust*100*n_correct/n_total)/adjust
            best_percent = percent if percent > best_percent else best_percent
            print("Epoch " + str(epoch+1) +" evaluation accuracy : " + str(percent) + "%\n")
        
    return best_percent