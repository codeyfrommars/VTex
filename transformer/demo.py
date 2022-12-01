import cv2
import mediapipe as mp
from collections import deque
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import torch
import torch.nn as nn
import numpy as np
import random
from dataset import CrohmeDataset, START, PAD, END, collate_batch
from torch.utils.data import DataLoader
from torchvision import transforms
import multiprocessing
import transformer_vtex as tv
import matplotlib.pyplot as plt
import editdistance
from PIL import Image, ImageOps

# Update these to use different test dataset
gt_test = "./transformer/data/groundtruth_2016.tsv"
tokensfile = "./transformer/data/tokens.tsv"
root = "./transformer/data/test/2016/"
checkpoint_path = "./checkpoints_new1631"

draw_dir = "./mediapipe/screenshots/"
# input image size to the transformer model
TRANS_IMG_SIZE = (256, 256)
# BEAM_SIZE = 10

imgWidth = 256
imgHeight = 256

max_trg_length = 100

transformers = transforms.Compose(
    [
        # Resize so all images have the same size
        transforms.Resize((imgWidth, imgHeight)),
        transforms.ToTensor(),
        # normalize
        transforms.Normalize([0.5], [0.5])
    ]
)

# Global variables. Modified in "main" if statement
Model = None
Sos_idx = None
Eos_idx = None
Pad_idx = None
Device = None

# sending screnshotted canvas to our transformer model
def convert(img_dir, beam_size = 10):

    image = Image.open(img_dir)

    # Remove alpha channel
    image = image.convert("RGB").convert('L')
    image = transformers(image)

    src = torch.tensor(image, device=Device)
    src = src.unsqueeze(dim=1)

    with torch.no_grad():
        output = Model.beam_search(src, Pad_idx, Sos_idx, Eos_idx, beam_size)

        output_text = ""
        for i in output:
            if i.item() != Sos_idx and i.item() != Eos_idx:
                output_text = output_text + test_dataset.id_to_token[i.item()]
        print ("Output:   " + output_text)

# run mediapipe program
def run():

    wpoints = [deque(maxlen = 1024)]

    # These indexes will be used to mark position
    # of pointers in colour array
    white_index = 0

    # The colours which will be used as ink for
    # the drawing purpose
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0),
            (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    # Here is code for Canvas setup
    paintWindow = np.zeros((471, 636, 3)) + 255 

    # For webcam input:
    cap = cv2.VideoCapture(0)
    pause = point = 0
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        imageIndex = 0
        screenshotFlag = True
        enableDraw = True
        # screenshotWait = 45
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # print(results)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Initially set finger count to 0 for each cap
            fingerCount = 0

            if results.multi_hand_landmarks:
            #   print(results.multi_hand_landmarks)  
                for hand_landmarks in results.multi_hand_landmarks:
                # Get hand index to check label (left or right)
                    handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                    handLabel = results.multi_handedness[handIndex].classification[0].label

                    # Set variable to keep landmarks positions (x and y)
                    handLandmarks = []

                    # Fill list with x and y positions of each landmark
                    for landmarks in hand_landmarks.landmark:
                        handLandmarks.append([landmarks.x, landmarks.y])

                    
                    # Thumb: TIP x position must be greater or lower than IP x position, 
                    #   deppeding on hand label.
                    # if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    #   fingerCount = fingerCount+1
                    # elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    #   fingerCount = fingerCount+1

                    # Other fingers: TIP y position must be lower than PIP y position, 
                    #   as image origin is in the upper left corner.

                    # if all 4 fingers of your left hand (no thumb) are raised, then clear drawings
                    if handLabel == "Right":
                        if handLandmarks[12][1] < handLandmarks[10][1] and \
                        handLandmarks[16][1] < handLandmarks[14][1] and \
                        handLandmarks[20][1] < handLandmarks[18][1] and \
                        handLandmarks[8][1] < handLandmarks[6][1]:
                            paintWindow = np.zeros((471, 636, 3)) + 255  
                            wpoints = [deque(maxlen = 1024)]
                            white_index = 0
                            continue

                        # if handLandmarks[12][1] >= handLandmarks[10][1] and \
                        # handLandmarks[16][1] >= handLandmarks[14][1] and \
                        # handLandmarks[20][1] >= handLandmarks[18][1] and \
                        # handLandmarks[8][1] < handLandmarks[6][1]:
                        #     enableDraw = True
                        # else:
                        #     enableDraw = False

                    elif handLabel == "Left":

                        # screenshot if 4 right fingers (no thumb) are raised
                        if handLandmarks[12][1] < handLandmarks[10][1] and \
                        handLandmarks[16][1] < handLandmarks[14][1] and \
                        handLandmarks[20][1] < handLandmarks[18][1] and \
                        handLandmarks[8][1] < handLandmarks[6][1] and \
                        screenshotFlag:
                            # resize canvas to Transformer image size (256, 256)
                            # paintImg = cv2.resize(paintWindow, TRANS_IMG_SIZE)
                            paintImg = paintWindow
                            canvasDir = draw_dir + str(imageIndex) + '.png'
                            cv2.imwrite(canvasDir,cv2.flip(paintImg, 1))
                            imageIndex += 1
                            screenshotFlag = False
                            # screenshotWait = 0
                            print("Saved Screenshot at", canvasDir)
                            convert(canvasDir)
                        

                        # right index finger is raised to draw (only if the 
                        # right index finger is raised, other right fingers can't be raised)
                        elif handLandmarks[12][1] >= handLandmarks[10][1] and \
                        handLandmarks[16][1] >= handLandmarks[14][1] and \
                        handLandmarks[20][1] >= handLandmarks[18][1] and \
                        handLandmarks[8][1] < handLandmarks[6][1] and enableDraw:     
                            point = handLandmarks[8]
                            point = (int(point[0]*image.shape[1]), int(point[1]*image.shape[0]))
                            if colorIndex == 0:
                                wpoints[white_index].appendleft(point)
                            screenshotFlag = True
                            # screenshotWait += 1
                        
                        # add empty point so the new point doesn't connect to the 
                        # previous point if paused for a long time
                        else:
                            wpoints.append(deque(maxlen = 512))
                            white_index += 1
                
                # draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
            # draw points (line segments between consecutive points)
            points = [wpoints]
            for i in range(len(points)):
                
                for j in range(len(points[i])):
                    
                    for k in range(1, len(points[i][j])):
                        
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                            
                        cv2.line(image, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)



            # Show all the windows
            cv2.imshow("Tracking", cv2.flip(image, 1))
            cv2.imshow("Paint", cv2.flip(paintWindow, 1))

            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

if __name__ == "__main__":
    """
    code to test transformer
    """

    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a Crohme Dataset to get the <EOS>
    train_dataset = CrohmeDataset(gt_test, tokensfile, root=root, crop=False)
    trg_pad_idx = train_dataset.token_to_id[PAD]
    eos_idx = train_dataset.token_to_id[END]
    trg_vocab_size =  len(train_dataset.token_to_id)

    # Initialize model
    Model = tv.Transformer(Device, trg_vocab_size, trg_pad_idx, max_trg_length, imgHeight, imgWidth).to(Device)

    # load test dataset
    test_dataset = CrohmeDataset(
        gt_test, tokensfile, root=root, crop=False, transform=transformers
    )
    Sos_idx = test_dataset.token_to_id[START]
    Eos_idx = test_dataset.token_to_id[END]
    Pad_idx = test_dataset.token_to_id[PAD]
    print("Loaded Test Dataset")

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=Device)
    Model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded Checkpoint")

    Model.eval()

    # run mediapipe
    run()