import os, shutil, pickle, re, time
import cv2, math

import pandas as pd
import numpy as np

from glob import glob
from zipfile import ZipFile
from tqdm import tqdm
from datetime import datetime

tqdm.pandas()



def cleaner(word):
    # word = re.sub('\[\'|\'\]', '', word).strip()
    word = re.sub('\[\'|\'\]|\'', '', word).strip()
    word = re.sub(r'\\n', '', word)

    return word


def clean_txt(df):
    df['word'] = df['word'].progress_apply(lambda x: cleaner(x)) # 리스트 -> 문자열로 바꿔주기
    return df


def check_and_mkdir(path_, rebuild=False):
    if rebuild:
        if os.path.exists(path_): shutil.rmtree(path_)
        os.mkdir(path_)
    else:
        if not os.path.exists(path_): os.mkdir(path_)


def resizer(img, resize, padding=True):
    # resize 기준 찾기: 가로 세로 중 더 큰 곳을 기준으로 resize 진행. 나머지는 비율 맞춰서 따라간다.
    who_is_longer = img.shape[1] > img.shape[0]
    if who_is_longer:   ratio = resize / img.shape[1]
    else:               ratio = resize / img.shape[0]
    
    # resize
    img = cv2.resize(img, dsize=(0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    
    # padding
    if padding:
        w, h = img.shape[1], img.shape[0]
        
        dw = (resize - w)/2
        dh = (resize - h)/2
        
        M = np.float32([[1, 0, dw], [0, 1, dh]])
        img = cv2.warpAffine(img, M, (resize, resize)) # 이동변환
    
    return img



def parse_landmark(result, keypoint_data, get_pose=True, get_lhand=False, get_rhand=False, spherical=False):
    '''
    result: holistic.process(image)
    '''
    if get_pose: # 키포인트 추출 - 몸통
        FLAG = 'pose'
        keypoint_data = parse_oneof_landmark(keypoint_data, result.pose_landmarks, FLAG, spherical)
    
    if get_lhand: # 키포인트 추출 - 왼손
        FLAG = 'lhand'
        keypoint_data = parse_oneof_landmark(keypoint_data, result.left_hand_landmarks, FLAG, spherical)
    
    if get_rhand: # 키포인트 추출 - 오른손
        FLAG = 'rhand'
        keypoint_data = parse_oneof_landmark(keypoint_data, result.right_hand_landmarks, FLAG, spherical)
    
    
    return keypoint_data
        
        


def parse_oneof_landmark(keypoint_data, body_part, FLAG, spherical):
    '''
    keypoint data: 딕녀서리 
    '''
    if body_part:
        for index, landmark in enumerate(body_part.landmark):
            part_x = f'{FLAG}_x{index}'
            part_y = f'{FLAG}_y{index}'
            part_z = f'{FLAG}_z{index}'
            
            x = landmark.x
            y = landmark.y
            z = landmark.z
                
            if not spherical: # original
                if part_x in keypoint_data: keypoint_data[part_x].append(x)
                else                      : keypoint_data[part_x] = [x]
                
                if part_y in keypoint_data: keypoint_data[part_y].append(y)
                else                      : keypoint_data[part_y] = [y]

                if part_z in keypoint_data: keypoint_data[part_z].append(z)
                else                      : keypoint_data[part_z] = [z]
    
            else: # Sperical 
                part_r = f'{FLAG}_r{index}'
                part_theta = f'{FLAG}_theta{index}'
                part_pi = f'{FLAG}_pi{index}'
                
                r = math.sqrt(x**2 + y**2 + z**2)
                theta = math.acos(z/r)
                pi = math.atan(y/x)
                
                
                if part_r in keypoint_data: keypoint_data[part_r].append(r)
                else                      : keypoint_data[part_r] = [r]
                
                if part_theta in keypoint_data: keypoint_data[part_theta].append(theta)
                else                      : keypoint_data[part_theta] = [theta]

                if part_pi in keypoint_data: keypoint_data[part_pi].append(pi)
                else                      : keypoint_data[part_pi] = [pi]
                
    return keypoint_data
    

class ExtractKeypoint:
    def __init__(self):
        self.import_libs()
        self.mp_holistic = self.mp.solutions.holistic
    
    
    def import_libs(self):
        try:
            import mediapipe as mp
            self.mp = mp
        except ModuleNotFoundError:
            raise ModuleNotFoundError("try: pip install mediapipe")
        
        try:
            import cv2
        except ModuleNotFoundError:
            raise ModuleNotFoundError("try: pip install opencv-python")
        
        
    def process_video(self, file_path):
        cap = cv2.VideoCapture(file_path)

        keypoint_data = []
        frame_number = 0

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=False) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                    
                # RGB 정렬
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(image)

                pose_data, lhand_data, rhand_data = [], [], []

                pose_data = self.parse_one_landmark(result.pose_landmarks, pose_data, 33)
                lhand_data = self.parse_one_landmark(result.left_hand_landmarks, lhand_data, 21)
                rhand_data = self.parse_one_landmark(result.right_hand_landmarks, rhand_data, 21)

                row_data = pose_data + lhand_data + rhand_data
                        
                video_name = os.path.basename(file_path) # 파일명.확장자 추출
                frame_rate = cap.get(cv2.CAP_PROP_FPS) # FPS
                time_per_frame = 1.0 / frame_rate # 1/FPS
                current_time = round(frame_number * time_per_frame, 3) # 시간정보 = 프레임수 * 1/FPS
                row_data.extend([video_name, str(current_time), str(frame_number)])
                keypoint_data.append(row_data)

                frame_number += 1
        cap.release()

        return keypoint_data
    
    
    def parse_one_landmark(self, part, data: list, num_landmarks: int):
        if part:
            for landmark in part.landmark:
                data.extend([str(landmark.x), str(landmark.y), str(landmark.z)])
        else:
            # landmarks 없을 때, 공란으로 처리
            for _ in range(num_landmarks * 3) :
                data.extend(["", "", ""])
                if len(data) >=num_landmarks * 3 :
                    break
        
        return data

    def get_keypoint_pkl(self, video_file, keypoints_dir):
        
        processed = self.process_video(video_file)
        filename = video_file.split('/')[-1].rstrip('.mp4')

        column_names = []
        for i in range(75):
            column_names.extend([f"x{i}", f"y{i}", f"z{i}"])

        column_names.extend(["filename", "time", "frame"])

        df = pd.DataFrame(processed, columns=column_names)

        with open(f"{keypoints_dir}/{filename}.pkl", 'wb') as f:
            pickle.dump(df, f)        
            
    def get_video_path(directory):
            return glob(f"{directory}/**/*.mp4", recursive=True)