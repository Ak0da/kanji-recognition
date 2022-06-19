print("Loading...")

from win32api import GetKeyState
from win32con import VK_LBUTTON, VK_CONTROL
from win32gui import GetCursorPos
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

detectionModel = kanji_detector().to(device=device).cpu()
#detectionModel.load_state_dict(torch.load('./Models/kanji_model_98_1.pth'))
detectionModel.load_state_dict(torch.load('./Models/kanji_model_top5_99_pertub.pth'))

def sortAccumulator(item):
    return item[1]

def getCaptures() -> list:
    capture_sizes = [20,40,60]
    half_capture_sizes = [capture_size//2 for capture_size in capture_sizes]
    images = []
    
    for i in range(len(capture_sizes)):

        half_capture_size = half_capture_sizes[i]

        posClick = GetCursorPos()
        rect = (posClick[0]-half_capture_size, 
                posClick[1]-half_capture_size,
                posClick[0]+half_capture_size,
                posClick[1]+half_capture_size)
        image = ImageGrab.grab(bbox=rect)
        images.append(image)
    return images

def identifySymbol(top_k: int, images: Image):
    
    
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
        
        list_tensors = [convert_tensor(image) for image in images]
        tensor_images = torch.stack(list_tensors, dim=0)

        results = activeModel(tensor_images.to(device=device))
        results = torch.nn.functional.softmax(results, dim=-1)

        results = results.sum(dim=0) #sums results for all batches
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
            
        acc_list = list(accumulator.items())
        acc_list.sort(key=sortAccumulator,reverse=True)
        clean_list = [item[0] for item in acc_list]
        
        del activeModel
        del tensor_images
        
        return clean_list[:top_k]
        

def isPressed(key) -> bool:
    return (GetKeyState(key) & ~1) != 0 # Left button down = 0 or 1. Button up = -127 or -128

def check_int(string: str):
    return re.match(r"[-+]?\d+(\.0*)?$", str) is not None

def validate_selection(selection: list) -> str:
    temp_str = "|"
    separator = ""
    for i in range(len(selection)):
        temp_str += selection[i] + ":" + str(i+1) + "|"
        separator += "_____"
    
    print(separator)
    print(temp_str)
    print(separator)
    print("Press 0 to skip")
    valid = False
    while not valid:
        answer = input("Selection : ")
        valid = re.match(r"[-+]?\d+(\.0*)?$", answer) is not None
        if valid:
            valid = int(answer) >= 0 and int(answer) <= len(selection)
        if not valid:
            print("Invalid selection. Retry.")
    idx = int(answer)-1
    if idx < 0:
        print("Skipped")
        return ""
    else:
        kanji = selection[idx]
        print(kanji + " selected\n")
        return kanji


state_left = isPressed(VK_LBUTTON)
state_ctrl = isPressed(VK_CONTROL)

images_captured = []

print("Ready to capture")
while True:
    a = isPressed(VK_LBUTTON)
    state_ctrl = isPressed(VK_CONTROL)
    
    if a != state_left: # Button state changed
        state_left = a
            
        if a and state_ctrl:
            isHolding = True
            images = getCaptures()
            images_captured.append(images) #Yes, list of lists
            print("Kanji " + str(len(images_captured)) + " selected")
        
    if not state_ctrl and len(images_captured) > 0:
        print("Selected " + str(len(images_captured)) + " kanjis")
        kanjis = ""
        for images in images_captured:
            selection = identifySymbol(top_k=5, images=images)
            kanji = validate_selection(selection)
            kanjis += kanji
            
        if len(kanjis) > 0:
            print("Searching for "+kanjis)
            webbrowser.open("https://jisho.org/search/"+kanjis, new=2, autoraise=True)
            
        images_captured.clear()
        print("\n\nReady to capture")