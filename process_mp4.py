import cv2
import numpy as np
import mediapipe as mp
import json
import os

warnings_only = True

def landmark_to_tuple(lm):
    return tuple([round(field[1],3) for field in lm.ListFields()])

def landmark_list_to_array(lm_list):
    return [landmark_to_tuple(lm) for lm in list(lm_list.ListFields()[0][1])]

def process_mp4(files, output, step=15):
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.2,
                                     min_tracking_confidence=0.05)

    output_file = open(output, 'a')

    for filepath in files:
        cap = cv2.VideoCapture(filepath)
        
        video_results = {'filename': filepath, 'output': []}

        counter = 0
        while cap.isOpened():
            try:
                # read only every nth frame
                if counter % step != 0:
                    break
                counter+=1
                
                # Try to read frame from video capture
                success, frame = cap.read()
                if not success:
                    print('Failed to read %s' % filepath)
                    break

                # process frame
                frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                                   interpolation = cv2.INTER_CUBIC)
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = hands.process(frame)

                # add frame data to video data
                video_results['output'].append(landmark_list_to_array(results.multi_hand_landmarks[0]))
            except:
                print('Failed to read frame')
                video_results['output'].append([])

        if len(video_results['output']) > 0:
            json.dump(video_results, output_file)
            if not warnings_only:
                print('Successfully processed and saved %s' % filepath)
        else:
            print('No frames processed for %s' % filepath)

def get_file_list(path):
    file_list = []
    for root, directories, files in os.walk(path, topdown=False):
        file_list += [(root + '/' + file) for file in files]
    return file_list

process_mp4(get_file_list('data/HandWashDataset'), 'preprocessed_data.json')
