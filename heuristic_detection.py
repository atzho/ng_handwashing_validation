import cv2
import mediapipe as mp
import math, time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def landmark_to_tuple(lm):
    return tuple([round(field[1],3) for field in lm.ListFields()])

def landmark_list_to_array(lm_list):
    return [landmark_to_tuple(lm) for lm in list(lm_list.ListFields()[0][1])]

def mean(joints):
  out = [0,0,0]
  for joint in joints:
    out[0] += joint[0]
    out[1] += joint[1]
    out[2] += joint[2]
  out[0] /= len(joints)
  out[1] /= len(joints)
  out[2] /= len(joints)
  return out

def SD(joints):
  avg = mean(joints)
  diff = 0
  for joint in joints:
    diff += math.sqrt((avg[0] - joint[0])**2 + (avg[1] - joint[1])**2 + (avg[2] - joint[2])**2)
  return diff

last_wash = time.time()
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
  min_detection_confidence=0.7,
  min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #hold label for 0.5 seconds
    if time.time() - last_wash > 0.5:
        label = 'None'
    if results.multi_hand_landmarks:
      if len(results.multi_hand_landmarks) == 2:
        left = landmark_list_to_array(results.multi_hand_landmarks[0])
        right = landmark_list_to_array(results.multi_hand_landmarks[1])
        if SD([mean(left), mean(right)])**2 / (SD(left) + SD(right)) < 0.01:
          label = 'Washing'
          last_wash = time.time()
        print(SD([mean(left), mean(right)])**2 / (SD(left) + SD(right)))
      for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    image = cv2.putText(image, label, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
