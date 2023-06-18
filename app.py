# Import all necessary packages
import numpy as np
import pyautogui
import cv2
from mss import mss
from itertools import count

dino = cv2.imread("./images/dino/start.png")
gg = cv2.imread("./images/screen/gg.png")

w, h = dino.shape[:-1]

DETECT_GAME_THRESHOLD = 0.75
JUMP_THRESHOLD = 240

IMAGE_GEN_COUNTER = count(0)

with mss() as sct:
    while True:
        sct_image = np.array(sct.grab(sct.monitors[0]))

        # Detect if there is a dino inside
        res = cv2.matchTemplate(
            cv2.cvtColor(sct_image, cv2.COLOR_RGB2GRAY), 
            cv2.cvtColor(dino, cv2.COLOR_RGB2GRAY), 
            cv2.TM_CCOEFF_NORMED
        )
        
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)

        if (maxVal >= DETECT_GAME_THRESHOLD):
            pyautogui.keyDown("enter")
            pyautogui.keyUp("enter")
            break
    
    i = 0
    while True:
        sct_image = np.array(sct.grab(sct.monitors[0]))

        game_over = cv2.matchTemplate(
            cv2.cvtColor(sct_image, cv2.COLOR_RGB2GRAY), 
            cv2.cvtColor(gg, cv2.COLOR_RGB2GRAY), 
            cv2.TM_CCOEFF_NORMED
        )

        (_, maxVal, _, _) = cv2.minMaxLoc(game_over)

        if (maxVal > DETECT_GAME_THRESHOLD): 
            break
        else: 
            (checkX, checkY) = maxLoc
            endX = checkX + dino.shape[1]
            endY = checkY + dino.shape[0]
            region = {"top": checkY, "left": checkX + w, "width": w, "height": h}
            sct_image = np.array(sct.grab(region))
            cv2.imwrite(f'./images/{next(IMAGE_GEN_COUNTER)}.png', sct_image)
            if (np.mean(sct_image) < JUMP_THRESHOLD):
                pyautogui.keyDown("up")
                pyautogui.keyUp("up")
