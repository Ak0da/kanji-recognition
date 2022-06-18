import win32api
import win32con
import win32gui
import time
import torch
from torchvision import transforms
from kanji_detection_model import kanji_detector
import json
from PIL import ImageGrab, Image

import re
import webbrowser


num2kanji_dict = {}
with open('num2kanji_dict.json', 'r') as f:
  num2kanji_dict = json.load(f)


device = "cuda" if torch.cuda.is_available() else "cpu"

detectionModel = kanji_detector().to(device=device)
detectionModel.load_state_dict(torch.load('./Models/kanji_model_98_1.pth'))

def sortAccumulator(item):
    return item[1]

def identifySymbol(top_k: int = 5):
    
    capture_sizes = [20,40,60]
    half_capture_sizes = [capture_size//2 for capture_size in capture_sizes]
    
    convert_tensor = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(64),
        transforms.ToTensor()
    ])
    
    accumulator = {}
    
    with torch.no_grad():
        activeModel = detectionModel.to(device=device)
        activeModel.eval()
        
        maxKanji = u"魔"
        maxValue = -9999999999.0
        
        for i in range(len(capture_sizes)):
            
            half_capture_size = half_capture_sizes[i]
            
            posClick = win32gui.GetCursorPos()
            rect = (posClick[0]-half_capture_size, 
                    posClick[1]-half_capture_size,
                    posClick[0]+half_capture_size,
                    posClick[1]+half_capture_size)
            image = ImageGrab.grab(bbox=rect)
            
            tensor_image = convert_tensor(image)
            
            results = activeModel(tensor_image.unsqueeze(0).to(device=device))
            #results = torch.nn.functional.softmax(results, dim=-1)
            
            
            #top_values, most_likely_indices = results.squeeze().topk(k=top_k,dim=0)
            top_values, most_likely_indices = results.squeeze().topk(k=results.size(-1),dim=0)
            selections = (most_likely_indices).tolist()
            selections_probs = top_values.tolist()
            
            kanjis = [num2kanji_dict[str(k)] for k in selections]
            #print(kanjis)
            
            for i in range(len(selections)):
                k = num2kanji_dict[str(selections[i])]
                v = selections_probs[i]
                
                if maxValue < v:
                    maxKanji = k
                    maxValue = v
                
                accumulator[k] = accumulator.get(k,0) + v
            
        #print("")
        acc_list = list(accumulator.items())
        acc_list.sort(key=sortAccumulator,reverse=True)
        clean_list = [item[0] for item in acc_list]
        #print(maxKanji)
        #print(clean_list[:20])
        
        
        print("__________________________________________________")
        
        del activeModel
        del tensor_image
        
        return clean_list[:10]
        

def isPressed(key) -> bool:
    return (win32api.GetKeyState(key) & ~1) != 0 # Left button down = 0 or 1. Button up = -127 or -128

def check_int(string: str):
    return re.match(r"[-+]?\d+(\.0*)?$", str) is not None

state_left = isPressed(win32con.VK_LBUTTON)
state_ctrl = isPressed(win32con.VK_CONTROL)

while True:
    a = isPressed(win32con.VK_LBUTTON)
    state_ctrl = isPressed(win32con.VK_CONTROL)
    
    if a != state_left: # Button state changed
        state_left = a
            
        if a and state_ctrl:
            isHolding = True
            probable_list = identifySymbol(top_k=5)
            
            temp_str = "|"
            for i in range(len(probable_list)):
                temp_str += probable_list[i] + ":" + str(i+1) + "|"
            
            print(temp_str)
            print("Press 0 to skip")
            valid = False
            while not valid:
                answer = input("Selection : ")
                valid = re.match(r"[-+]?\d+(\.0*)?$", answer) is not None
                if valid:
                    valid = int(answer) >= 0 and int(answer) <= len(probable_list)
                if not valid:
                    print("Invalid selection. Retry.")
            idx = int(answer)-1
            if idx < 0:
                print("Skipped")
            else:
                kanji = probable_list[idx]
                print(kanji + " selected\n")
                webbrowser.open('https://jisho.org/search/'+kanji, new=2)