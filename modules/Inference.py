import os, sys, pickle
import cv2
import math
import pandas as pd
import numpy as np
from collections import OrderedDict

from glob import glob
from tqdm import tqdm
from easydict import EasyDict as ed

import mediapipe as mp
import torch
from torch import nn
tqdm.pandas()

torch.set_default_dtype(torch.float64)

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()
container = os.environ.get('container')

# Keypoint 추출
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

    def get_keypoint_pkl(self, video_file, keypoints_dir=None, save=False):
        
        processed = self.process_video(video_file)
        filename = video_file.split('/')[-1].rstrip('.mp4')

        column_names = []
        for i in range(75):
            column_names.extend([f"x{i}", f"y{i}", f"z{i}"])

        column_names.extend(["filename", "time", "frame"])

        df = pd.DataFrame(processed, columns=column_names)

        if save:
            with open(f"{keypoints_dir}/{filename}.pkl", 'wb') as f:
                pickle.dump(df, f)        
        
        return df
            
    def get_video_path(directory):
            return glob(f"{directory}/**/*.mp4", recursive=True)


# Augmentation
class CustomAugmentation:
    def __init__(self, num_landmarks=23, maxlen=70):
        self.num_mask_range = 0
        self.num_landmarks = num_landmarks
        self.coor_dim = 2
        self.maxlen = maxlen


    def act(self, x): #numpy.array -> torch.tensor
        X_data = self.transform(x)
        X_data = torch.tensor(X_data)
        return torch.tensor(X_data)



    def transform(self, x): # numpy
        x = self.distance_normalization(x)
        x = self.center_crop(x)
        return x



    def distance_normalization(self, x):
        x = x.reshape((-1, self.num_landmarks, self.coor_dim))
        '''holistic 상반신 좌표 설명
        0~22 : 상반신
            - 0: 코(ref)
                - 0~10 : 얼굴
            - 13 : 오른쪽 팔꿈치(ref)
                - 11, 13, 15 : 오른쪽 팔
            - 14 : 왼쪽 팔꿈치(ref)
                - 12, 14, 16 : 왼쪽 팔
            - 15 ~ 22 : 양손 (버리기)
        '''
        # 0~22 / 23~43 /  44~64
        # center
        center              = np.mean(x, axis=1)

        # referece points
        nose_ref            = x[:, 0]
        right_elbow_ref     = x[:, 13]
        left_elbow_ref      = x[:, 14]
        left_wrist_ref      = x[:, 23]
        right_wrist_ref     = x[:, 44]

        # Partition of landmarks
        face_section        = x[:, :11] # 얼굴
        right_arm_section   = x[:, 11:16:2] # 오른팔
        left_arm_section    = x[:, 12:17:2] # 왼팔
        two_hands_section   = x[:, 15:23] # 버릴 좌표
        right_hand_section  = x[:, 44:64] # 오른손
        left_hand_section   = x[:, 23:44] # 왼손


        # 대표거리구하기: center - reference point
        euclidean = lambda ax, ay: np.sqrt(((center[:, 0] - ax)**2 + (center[:, 1] - ay)**2) + 1e-9)

        d_nose          = euclidean(nose_ref[:, 0], nose_ref[:, 1])
        d_right_elbow   = euclidean(right_elbow_ref[:, 0], right_elbow_ref[:, 1])
        d_left_elbow    = euclidean(left_elbow_ref[:, 0], left_elbow_ref[:, 1])
        d_right_wrist   = euclidean(right_wrist_ref[:, 0], right_wrist_ref[:, 1])
        d_left_wrist    = euclidean(left_wrist_ref[:, 0], left_wrist_ref[:, 1])

        # Normalized Distance
        distance = lambda section, d_ref: (section - center.reshape(-1, 1, 2)) / (d_ref.reshape(-1, 1, 1) + 1e-9)

        d_face       = distance(face_section, d_nose)
        d_right_arm  = distance(right_arm_section, d_right_elbow)
        d_left_arm   = distance(left_arm_section, d_left_elbow)
        d_right_hand = distance(right_hand_section, d_right_wrist)
        d_left_hand  = distance(left_hand_section, d_left_wrist)

        # Rescale: MinMaxScaler
        def mm(arr):
            if len(arr) == 0:
                print(arr)
            arr_x, arr_y = arr[..., 0], arr[..., 1]
            
            mm_x = (arr_x - arr_x.min()) / (arr_x.max() - arr_x.min() + 1e-9)
            mm_y = (arr_y - arr_y.min()) / (arr_y.max() - arr_y.min() + 1e-9)

            mm_x = np.expand_dims(mm_x, axis=-1)
            mm_y = np.expand_dims(mm_y, axis=-1)

            return np.concatenate([mm_x, mm_y], axis=-1)

        mm_face = mm(d_face)
        mm_right_arm = mm(d_right_arm)
        mm_left_arm = mm(d_left_arm)
        mm_right_hand = mm(d_right_hand)
        mm_left_hand = mm(d_left_hand)

        # 병합 및 reshape하여 반환
        result = np.concatenate([mm_face, mm_right_arm, mm_left_arm, mm_right_hand, mm_left_hand], axis=1)
        return result.reshape(-1, (self.num_landmarks-7) * self.coor_dim)
        # return result


    def center_crop(self, x):
        # setting
        self.num_crop = self.maxlen

        # set index
        start_idx = 5 
        end_idx = start_idx + self.num_crop

        zero_pad = np.zeros((100, (self.num_landmarks-7) * self.coor_dim))
        x = np.concatenate([x, zero_pad], axis=0)

        # crop
        return x[start_idx:end_idx, :]



    
    
# Dataset
class TestDS(torch.utils.data.Dataset):
    def __init__(self, videopath, device=None):
        global container
        super().__init__()
        self.device = device
        
        self.word2idx = self.load_word2idx() # {단어 : 라벨}
        self.num_classes = len(self.word2idx) # 클래스 수
        
        self.videopath = videopath
        

        # self.num_landmarks = 23 # pose(상반신)
        # self.num_landmarks = 33 # pose
        self.num_landmarks = 65 # holistic(하반신 제거)
        # self.num_landmarks = 75 # holistic
        self.ek = ExtractKeypoint()
        self.CA = CustomAugmentation(self.num_landmarks)
        
        
        
    def __len__(self):
        return 1


    def __getitem__(self, idx):
        filename = self.videopath.split('/')[-1]
        # print(f"파일명: {filename}")
        # 키포인트 추출
        keypoints = self.extract_keypoint(self.videopath)
        
        X_tensor = self.CA.act(keypoints)
        return X_tensor.to(self.device), 0
        

        
    def load_word2idx(self):
        with open(f'{container}/data/word2idx.pkl', 'rb') as f: # 165개 단어(의료, 일상)
            word2idx = pickle.load(f)
        
        return word2idx
    
    
    def extract_keypoint(self, videopath):
        # 키포인트 추출
        keypoints = self.ek.get_keypoint_pkl(videopath, save=False)
        keypoints = keypoints.iloc[:, :-3]

        # 하반신 컬럼
        legs = ['x23', 'y23', 'z23', 'x24', 'y24', 'z24', 'x25', 'y25', 'z25', 'x26',
                'y26', 'z26', 'x27', 'y27', 'z27', 'x28', 'y28', 'z28', 'x29', 'y29',
                'z29', 'x30', 'y30', 'z30', 'x31', 'y31', 'z31', 'x32', 'y32', 'z32']


        # keypoints = keypoints # 75개 모두 사용하는 경우
        # keypoints = keypoints.iloc[:, :69] # pose - 상반신만 사용하는 경우 -> 23 * 3 = 69
        # keypoints = keypoints.iloc[:, :99] # pose 모두 사용하는 경우
        keypoints = keypoints.drop(legs, axis=1, inplace=False) # holistic에서 하반신만 버릴 때
        
        keypoints = keypoints.replace('', 0.0).dropna(axis=0) # 0으로 대체
        keypoints = keypoints.to_numpy().astype(np.float64)
        keypoints = keypoints.reshape(-1, self.num_landmarks, 3)[..., :2] # x, y, z -> x, y
        
        return keypoints
        


def my_collate_fn(samples, is_graph=True):
    ''' augmentation하면서 뻥튀기 된 batch를 다시 정렬해준다.
    '''
    
    X_collate = torch.stack([sample[0] for sample in samples])
    y_collate = torch.stack([sample[1] for sample in samples])
    
    X_collate = torch.reshape(X_collate, (-1, 70, (test_ds.num_landmarks-7) * 2))
    y_collate = torch.reshape(y_collate, (-1,))
    return (X_collate, y_collate)




# Model
class PositionalEncoding(nn.Module):
    '''reference:
    https://ysg2997.tistory.com/11
    - customized
    '''

    def __init__(self, dim_model, max_len, device):
        super().__init__()
        self.device =device
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model, device=device)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        # pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # print(f"token_embedding: {token_embedding.shape}")
        # print(f"pos_encoding: {self.pos_encoding[:token_embedding.size(0), :].shape}")

        #  pos encoding
        return token_embedding + self.pos_encoding[:token_embedding.size(0), :]



class Model(torch.nn.Module):
    def __init__(self, num_classes, transformer_dropout, maxlen, device):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.embedding = nn.Sequential(OrderedDict([
            ('Linear_Embedding', nn.Linear(in_features=(test_ds.num_landmarks-7) * 2, out_features=512, device=device)),
            ('Positional_Encoding', PositionalEncoding(dim_model=512, max_len=maxlen, device=device))
        ]))
        # self.embedding = nn.Linear(in_features=66, out_features=512, device=device)
        self.backbone = nn.Sequential(OrderedDict([
            ('BN_1', nn.BatchNorm1d(num_features=maxlen + 1, eps=1e-5, momentum=0.1, affine=True, device=device)),
            ('Transformer_1', nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=transformer_dropout,
                                                         batch_first=True, device=device)),
            ('BN_2', nn.BatchNorm1d(num_features=maxlen + 1, eps=1e-5, momentum=0.1, affine=True, device=device)),
            ('Transformer_2', nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=transformer_dropout,
                                                         batch_first=True, device=device)),
            ('BN_3', nn.BatchNorm1d(num_features=maxlen + 1, eps=1e-5, momentum=0.1, affine=True, device=device)),
            ('Transformer_3', nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=transformer_dropout,
                                                         batch_first=True, device=device)),
            ('BN_4', nn.BatchNorm1d(num_features=maxlen + 1, eps=1e-5, momentum=0.1, affine=True, device=device)),
            ('Transformer_4', nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=transformer_dropout,
                                                         batch_first=True, device=device)),
            ('BN_5', nn.BatchNorm1d(num_features=maxlen + 1, eps=1e-5, momentum=0.1, affine=True, device=device)),
            ('Transformer_5', nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=transformer_dropout,
                                                         batch_first=True, device=device)),


        ]))
        

    def forward(self, x):
        x = self.embedding(x)

        ##### cls token #####
        batch_size = x.size(0)
        emb_dim = x.size(-1)
        cls_token = torch.ones((batch_size, 1, emb_dim), device=self.device)
        x = torch.cat([cls_token, x], dim=1)
        ##### cls token #####

        output = self.backbone(x)
        # return output
        return output[:, 0]
        # return self.classifier(output[:, 0])



class TLModel(torch.nn.Module):
    def __init__(self, num_classes, transformer_dropout, maxlen, device):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        self.pretrained = Model(76, transformer_dropout, maxlen, device)
        self.pretrained.classifier = nn.Identity()
        self.classifier = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features=512, out_features=num_classes, device=device)),
            ('Softmax', nn.Softmax(dim=-1))
        ]))
    
    def forward(self, x):
        x = self.pretrained(x)
        return self.classifier(x)


# Predict
class Predict:
    def __init__(self, device):
        global container
        weight_pt = f'{container}/model/SSLRv2/best.pt'
        
        self.device = device

        # num_classes, idx2word
        global test_ds
        test_ds = TestDS('', device)
        num_classes = test_ds.num_classes
        self.idx2word = {idx:word for word, idx in test_ds.word2idx.items()}

        # Model
        self.model = TLModel(num_classes, 0.25, 70, self.device)
        self.model.load_state_dict(torch.load(weight_pt, map_location=self.device), strict=False)
        self.model.eval()

    def predict(self, videopath, top1=True):
        test_ds = TestDS(videopath, self.device)
        X_data = test_ds[0][0]
        
        # Predict
        y_pred = self.model(X_data)
        
        top_1 = y_pred.argmax()
        top_1 = int(top_1)
        top_1_word = self.idx2word[top_1]
        
        top_5 = y_pred.argsort().tolist()[::-1][0][:5]
        top_5_words = [self.idx2word[el].split('_')[0] for el in top_5]
        
        if top1:
            return top_1_word
        else:
            return top_5_words