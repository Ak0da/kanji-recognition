import torch
import PySimpleGUI as sg
import re
import webbrowser
import json

from PIL import ImageGrab, Image
from win32api import GetKeyState
from win32con import VK_LBUTTON, VK_CONTROL
from win32gui import GetCursorPos
from torchvision import transforms
from kanji_detection_model import kanji_detector

isRunning = True
device = "cuda" if torch.cuda.is_available() else "cpu"

layout = [
            [sg.Push(), sg.Text("Select kanji(s)", key="Instructions"), sg.Push()], 
    
            [sg.Push(), 
             sg.Button("魔", key="0", size=(3,0), font=("Helvetica", 50)), 
             sg.Button("", key="1", size=(3,0), font=("Helvetica", 50)), 
             sg.Button("", key="2", size=(3,0), font=("Helvetica", 50)), 
             sg.Button("", key="3", size=(3,0), font=("Helvetica", 50)), 
             sg.Button("", key="4", size=(3,0), font=("Helvetica", 50)), 
             sg.Push()],
    
             [sg.Push(), sg.Text("Currently selected : ", key="Selection", font=("Helvetica", 20)), sg.Push()], 
    
             [sg.Push(), 
             sg.Button("Skip this kanji", key="Skip"), 
             sg.Button("Retake this kanji", key="Retake"), 
             sg.Button("Cancel all selection", key="Cancel"),
             sg.Push()]
             ,
             [sg.Push(),
             sg.Button("Search", key="Search", size=(-1,0), font=("Helvetica", 20)),
             sg.Push()]
         ]

button_keys = ["0","1","2","3","4","Skip","Retake","Cancel","Search"]
button_kanji_keys = ["0","1","2","3","4"]

def setButtonsListInteractible(window1,button_list:list,enabled:bool):
     for key in button_list:
        window1[key].update(disabled=(not enabled))

def setButtonsInteractible(window1,enabled:bool):
    for key in button_keys:
        window1[key].update(disabled=(not enabled))

def setButtonsSymbol(window1,symbols:list):
    for i in range(len(button_kanji_keys)):
        window1[button_kanji_keys[i]].update(symbols[i])
        
def resetButtonTexts(window1):
    for key in button_kanji_keys:
        window1[key].update("")
        
def setInstructions(window1,message:str):
    window1["Instructions"].update(message)

def loadModel() -> torch.nn.Module:
    detectionModel = kanji_detector().to(device=device)
    detectionModel.load_state_dict(torch.load('./Models/kanji_model_v7_top5_96_eval.pth'))
    detectionModel(torch.rand((1,1,64,64)).float().to(device=device)) #Allocate memory in advance
    detectionModel = detectionModel.cpu() #Allocate memory in advance
    return detectionModel
        
def isPressed(key) -> bool:
    return (GetKeyState(key) & ~1) != 0 # Left button down = 0 or 1. Button up = -127 or -128

def check_int(string: str):
    return re.match(r"[-+]?\d+(\.0*)?$", str) is not None

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

def sortAccumulator(item):
    return item[1]

def identifySymbol(top_k: int, images: Image) -> list:
    
    convert_tensor = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(64,interpolation=transforms.InterpolationMode.BICUBIC),
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
        
def MultiCaptureLoop() -> list:
    global isRunning
    
    state_left = isPressed(VK_LBUTTON)
    state_ctrl = isPressed(VK_CONTROL)
    images_captured = []

    while True:
        event, values = window.Read(timeout = 20)
        if event == sg.WIN_CLOSED:
            isRunning = False
            break
        
        a = isPressed(VK_LBUTTON)
        state_ctrl = isPressed(VK_CONTROL)

        if a != state_left: # Button state changed
            state_left = a

            if a and state_ctrl:
                isHolding = True
                images = getCaptures()
                images_captured.append(images) #Yes, list of lists

        if not state_ctrl and len(images_captured) > 0:
            return images_captured
        
    return[]
        
def SingleCaptureLoop() -> list:
    global isRunning
    
    state_left = isPressed(VK_LBUTTON)
    state_ctrl = isPressed(VK_CONTROL)
    images_captured = []
    
    while True:
        event, values = window.Read(timeout = 20)
        if event == sg.WIN_CLOSED:
            isRunning = False
            break
        
        a = isPressed(VK_LBUTTON)
        state_ctrl = isPressed(VK_CONTROL)

        if a != state_left: # Button state changed
            state_left = a

            if a and state_ctrl:
                isHolding = True
                images = getCaptures()
                images_captured.append(images) #Yes, list of lists
                return images_captured
            
    return images_captured

kanjis = ""
list_images_captured = []
def identificationLoop(window1) -> bool:
    global kanjis
    global current_selection
    global isRunning
    
    kanji_selected = True
    setButtonsListInteractible(window,["Search"],False)
    
    while len(list_images_captured) > 0:
        
        setInstructions(window1,"Select the kanji you think you clicked on")
        
        if kanji_selected:
            kanji_selected = False
            kanji_list = identifySymbol(len(button_kanji_keys), list_images_captured.pop(0))
            setButtonsSymbol(window1,kanji_list)
        
        event, values = window1.Read()
        if event == sg.WIN_CLOSED:
            isRunning = False
            break
            
        if event == "Skip":
            kanji_selected = True #Fake, but pops the images as intended
        
        if event == "Retake":
            setInstructions(window1,"Select kanji somewhere on your screen")
            
            setButtonsInteractible(window,False)
            images_captured = SingleCaptureLoop()
            setButtonsInteractible(window,True)
            setButtonsListInteractible(window,["Search"],len(kanjis)>0)
            
            list_images_captured.insert(0, images_captured[0])
            kanji_selected = True
            
            if not isRunning:
                break
            
        try:
            index = int(event)
            if(index >= 0 and index <= 4):
                kanjis += window1[event].get_text()
                window1["Selection"].update("Currently selected : " + kanjis)
                setButtonsListInteractible(window,["Search"],True)
                kanji_selected = True
        except:
            pass
        
        if event == "Cancel":
            list_images_captured.clear()
            kanjis = ""
            window1["Selection"].update("Currently selected : None")
            return True
        
        if event == "Search" and len(kanjis) > 0:
            webbrowser.open("https://jisho.org/search/"+kanjis, new=2, autoraise=True)
            
    return False

            
            
            
# Create the window
window = sg.Window("Kanji reader", layout, element_padding=(0,5))
window.finalize()
setInstructions(window,"Program is loading, please wait")
window.force_focus()

num2kanji_dict = {}
with open('num2kanji_dict.json', 'r') as f:
  num2kanji_dict = json.load(f)


detectionModel = loadModel()

isRunning = True
isStartingCapture = True
setButtonsInteractible(window,False)
# Create an event loop
while isRunning:
    event, values = window.Read(timeout = 100)
    # End program if user closes window or
    # presses the OK button
    
    if event == sg.WIN_CLOSED:
        break
        
    if isStartingCapture:
        resetButtonTexts(window)
        setInstructions(window,"Select kanji(s) somewhere on your screen")
        isStartingCapture = False
        setButtonsInteractible(window,False)
        #print("In capture loop")
        
        images_captured = MultiCaptureLoop()
        list_images_captured.extend(images_captured)
        if not isRunning:
            break
            
        #print("Out capture loop")
        setButtonsInteractible(window,True)
        setInstructions(window,"Select the kanji you think you clicked on")
        wasCancelled = identificationLoop(window)
        isStartingCapture = wasCancelled
        setInstructions(window,"Capture complete")
        setButtonsListInteractible(window,["Skip","Retake","0","1","2","3","4"],False)
    
    if event == "Search":
        webbrowser.open("https://jisho.org/search/"+kanjis, new=2, autoraise=True)
        
    if event == "Cancel":
            list_images_captured.clear()
            kanjis = ""
            window["Selection"].update("Currently selected : None")
            isStartingCapture = True
            
    
window.close()