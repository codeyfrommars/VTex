import cv2
import mediapipe as mp
from collections import deque
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

bpoints = [deque(maxlen = 1024)]
gpoints = [deque(maxlen = 1024)]
rpoints = [deque(maxlen = 1024)]
ypoints = [deque(maxlen = 1024)]
wpoints = [deque(maxlen = 1024)]

# These indexes will be used to mark position
# of pointers in colour array
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
white_index = 0

# The colours which will be used as ink for
# the drawing purpose
colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0),
        (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((471, 636, 3))   

# For webcam input:
cap = cv2.VideoCapture(0)
pause = point = 0
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    imageIndex = 0
    screenshotFlag = True
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

                # Test conditions for each finger: Count is increased if finger is 
                #   considered raised.
                # Thumb: TIP x position must be greater or lower than IP x position, 
                #   deppeding on hand label.
                # if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                #   fingerCount = fingerCount+1
                # elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                #   fingerCount = fingerCount+1

                # Other fingers: TIP y position must be lower than PIP y position, 
                #   as image origin is in the upper left corner.

                # if all 5 fingers of your left hand are raised, then clear drawings
                if handLabel == "Right":
                    if handLandmarks[12][1] < handLandmarks[10][1] and \
                    handLandmarks[16][1] < handLandmarks[14][1] and \
                    handLandmarks[20][1] < handLandmarks[18][1] and \
                    handLandmarks[8][1] < handLandmarks[6][1]:
                        paintWindow = np.zeros((471, 636, 3))   
                        wpoints = [deque(maxlen = 1024)]
                        white_index = 0
                        continue

                elif handLabel == "Left":
                    # screenshot if 5 right fingers are raised
                    if handLandmarks[12][1] < handLandmarks[10][1] and \
                    handLandmarks[16][1] < handLandmarks[14][1] and \
                    handLandmarks[20][1] < handLandmarks[18][1] and \
                    handLandmarks[8][1] < handLandmarks[6][1] and \
                    screenshotFlag:
                        cv2.imwrite('/UTAustin/Fall2022/CV/VTex/mediapipe/screenshots/' + str(imageIndex) + '.png',cv2.flip(paintWindow, 1))
                        imageIndex += 1
                        screenshotFlag = False
                        # screenshotWait = 0
                        print("here")

                    # right index finger is raised to draw
                    elif handLandmarks[12][1] >= handLandmarks[10][1] and \
                    handLandmarks[16][1] >= handLandmarks[14][1] and \
                    handLandmarks[20][1] >= handLandmarks[18][1] and \
                    handLandmarks[8][1] < handLandmarks[6][1]:     
                        # index[0] = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width)
                        # index[0] = max(0, min(index[0], frame_width))
                        # index[1] = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height)
                        # index[1] = max(0, min(index[1], frame_height))
                        point = handLandmarks[8]
                        point = (int(point[0]*image.shape[1]), int(point[1]*image.shape[0]))
                        if colorIndex == 0:
                            wpoints[white_index].appendleft(point)
                        screenshotFlag = True
                        # screenshotWait += 1
                    else:
                        wpoints.append(deque(maxlen = 512))
                        white_index += 1
                        # screenshotFlag = True
                        # screenshotWait += 1
                # print(point)
            #   pause += 1
            #   if pause >= 10:
            #     print(handLandmarks[8][1], handLandmarks[6][1])
            #     pause = 0
            #   fingerCount = fingerCount+1

            # if colorIndex == 0:
            #         bpoints[blue_index].appendleft(center)
            #     elif colorIndex == 1:
            #         gpoints[green_index].appendleft(center)
            #     elif colorIndex == 2:
            #         rpoints[red_index].appendleft(center)
            #     elif colorIndex == 3:
            #         ypoints[yellow_index].appendleft(center)
                        
            # Append the next deques when nothing is
            # detected to avois messing up
            
            # bpoints.append(deque(maxlen = 512))
            # blue_index += 1
            # gpoints.append(deque(maxlen = 512))
            # green_index += 1
            # rpoints.append(deque(maxlen = 512))
            # red_index += 1
            # ypoints.append(deque(maxlen = 512))
            # yellow_index += 1

            # Draw lines of all the colors on the
            # canvas and frame
        # points = [bpoints, gpoints, rpoints, ypoints]
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
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



# Flip = 1 means the image is flipped horizontally for a selfie-view display.
# FLIP = 1

# if FLIP:
#     flippedImage = cv2.flip(image, 1)

#     # Display finger count
#     cv2.putText(flippedImage, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
#     cv2.imshow('MediaPipe Hands', flippedImage)

# else:
#     # Display finger count
#     cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
#     cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()