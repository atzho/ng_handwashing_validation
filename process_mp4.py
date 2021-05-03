import cv2
import numpy as np
import mediapipe as mp
import json
import os

# Author: Andrew Zhong
# Last Modified: 4/30/2021
# Data preprocessing stage for NG COVID-19 AI Challenge
# Processing hand action data into hand pose data using mediapipe

def landmark_to_tuple(lm):
    return tuple([round(field[1],3) for field in lm.ListFields()])

def landmark_list_to_array(lm_list):
    return [landmark_to_tuple(lm) for lm in list(lm_list.ListFields()[0][1])]

def process_mp4(files, output, step=15, debug_msgs = True, warnings_only = True, track_frames = False):
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.2,
                                     min_tracking_confidence=0.1)

    output_file = open(output, 'a')

    data = []
    file_num = 1
    frames_good = 0
    total_frames = 0

    for filepath in files:
        cap = cv2.VideoCapture(filepath)
        
        video_results = {'filename': filepath, 'output': []}

        counter = 0
        while cap.isOpened():
            # read only every nth frame
            success, frame = cap.read()
            if not success:
                break
            if counter % step == 0:
                total_frames += 1
                try:                    
                    # Try to read frame from video capture
    
                    # process frame
                    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                                       interpolation = cv2.INTER_CUBIC)
                    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False
                    results = hands.process(frame)
    
                    # add frame data to video data
                    joints = []
                    # check for hand count
                    if len(results.multi_hand_landmarks) != 0:
                        joints.append(landmark_list_to_array(results.multi_hand_landmarks[0]))
                        if len(results.multi_hand_landmarks) == 2:
                            frames_good += 1
                            joints.append(landmark_list_to_array(results.multi_hand_landmarks[1]))
                        video_results['output'].append(joints)
                    else:
                        video_results['output'].append([])
                        if debug_msgs:
                            print('Empty frame')
                except:
                    if debug_msgs:
                        print('Failed to read frame')
                    video_results['output'].append([])
                if track_frames:
                    print('%i/%i good frames' % (frames_good, total_frames))
            counter += 1

        if len(video_results['output']) > 0:
            data.append(video_results)
            if debug_msgs and not warnings_only:
                print('Successfully processed and saved %s' % filepath)
        else:
            if debug_msgs:
                print('No frames processed for %s' % filepath)
            
        print('%i of %i files processed...' % (file_num, len(files)))
        file_num += 1
        #return data#testing

    # only writes at end; if batch fails have to restart anyways :)
    json.dump(data, output_file)

    return total_frames

def get_file_list(path):
    file_list = []
    for root, directories, files in os.walk(path, topdown=False):
        file_list += [(root + '/' + file) for file in files]
    return file_list

if __name__ == "__main__":
    process_mp4(get_file_list('data/UCF101'), 'preprocessed_negatives.json')
