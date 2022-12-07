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
from dataset import CrohmeDataset, START, PAD, END, collate_batch
from torch.utils.data import DataLoader
from torchvision import transforms
import transformer_vtex as tv
import matplotlib.pyplot as plt
import editdistance
from PIL import Image, ImageOps

# Update these to use different test dataset
gt_test = "./transformer/data2/groundtruth_2019.txt"
tokensfile = "./transformer/tokens.txt"
root = "./transformer/data2/2019/"
checkpoint_path = "./checkpoints_bttr_data500"

draw_dir = "./mediapipe/screenshots/"
# input image size to the transformer model
TRANS_IMG_SIZE = (256, 256)
# BEAM_SIZE = 10

imgWidth = 256
imgHeight = 256

max_trg_length = 55

transformers = transforms.Compose(
    [
        # Resize so all images have the same size
        # transforms.Resize((1200, 700)),
        transforms.ToTensor(), # normalize to [0,1]
        # normalize
        # transforms.Normalize([0.5], [0.5])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    # image = Image.open("./transformer/data2/2019/ISICal19_1201_em_750.bmp")
    image = Image.open(img_dir)

    # Remove alpha channel
    image = image.convert("RGB").convert('L')
    ary = np.array(image)
    print(np.unique(ary))
    print(ary.shape)
    print(type(ary))
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

    pixelSize = 5

    points = []
    for i in range(pixelSize**2):
        points.append([deque(maxlen = 1024)])

    # These indexes will be used to mark position
    # of pointers in colour array
    white_index = 0

    # The colours which will be used as ink for
    # the drawing purpose
    colors = [255, (255, 0, 0), (0, 255, 0),
            (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    # Here is code for Canvas setup
    paintWindow = np.zeros((471, 636, 1))

    # dir = [(-1, -1), (0, -1), (1, -1), (0, -1), (0,0), (0, 1), (1,-1), (1,0), (1,1)]

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

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
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

                    # Other fingers: TIP y position must be lower than PIP y position, 
                    #   as image origin is in the upper left corner.

                    # if all 4 fingers of your left hand (no thumb) are raised, then clear drawings
                    if handLabel == "Right":
                        if handLandmarks[12][1] < handLandmarks[10][1] and \
                        handLandmarks[16][1] < handLandmarks[14][1] and \
                        handLandmarks[20][1] < handLandmarks[18][1] and \
                        handLandmarks[8][1] < handLandmarks[6][1]:
                            paintWindow = np.zeros((471, 636, 1)) 
                            for i in range(pixelSize**2):
                                points[i] = [deque(maxlen = 1024)]
                            white_index = 0
                            continue

                    elif handLabel == "Left":

                        # screenshot if 4 right fingers (no thumb) are raised
                        if handLandmarks[12][1] < handLandmarks[10][1] and \
                        handLandmarks[16][1] < handLandmarks[14][1] and \
                        handLandmarks[20][1] < handLandmarks[18][1] and \
                        handLandmarks[8][1] < handLandmarks[6][1] and \
                        screenshotFlag:
                            # resize canvas to Transformer image size (256, 256)
                            paintImg = paintWindow
                            canvasDir = draw_dir + str(imageIndex) + '.bmp'
                            paintImg = cv2.resize(paintImg, (300, 235), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(canvasDir,cv2.flip(paintImg, 1))
                            imageIndex += 1
                            screenshotFlag = False
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
                            count = 0
                            if colorIndex == 0:
                                for i in range(-(pixelSize//2), pixelSize//2+1):
                                    for j in range(-(pixelSize//2), pixelSize//2+1):
                                        print(count)
                                        points[count][white_index].appendleft((point[0]+j, point[1]+i))
                                        count += 1
                            screenshotFlag = True
                        
                        # add empty point so the new point doesn't connect to the 
                        # previous point if paused for a long time
                        else:
                            for i in range(pixelSize**2):
                                points[i].append(deque(maxlen=512))
                            white_index += 1
                
                # draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
            for i in range(len(points)):
                
                for j in range(len(points[i])):
                    
                    for k in range(1, len(points[i][j])):
                        
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                            
                        cv2.line(image, points[i][j][k - 1], points[i][j][k], colors[0], 2)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[0], 2)



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
    Model = tv.Transformer(Device, trg_vocab_size, trg_pad_idx, max_trg_length).to(Device)

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