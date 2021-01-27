# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:44:59 2021

@author: dominicdathan
"""

import argparse
import logging
import pathlib

import numpy as np

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# image handling
from PIL import Image

# plotting
import matplotlib.pyplot as plt


import cv2 

class Net(nn.Module):
    "Pytorch neural network model class for team and player predictions"
    def __init__(self):
        super(Net, self).__init__()
        # first layer is a Conv2d layer with 6 filters
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        
        # pooling to be applied after 1st and 2nd layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # second layer is a Conv2d layer with 16 filters
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        
        # third layer is a linear layer with 84 output features
        self.fc1 = nn.Linear(16 * 56 * 56, 84)
        
        # output layer for the first class (team)
        self.fc2 = nn.Linear(84, 6)
        
        # output layer for the second class (person)
        self.fc3 = nn.Linear(84,54)
        
        # dropout layer with probability 20% to be applied after the second conv2d layers
        # aims to prevent overfitting to the training set
        self.drop_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        # first layer and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # second layer and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # dropout layer
        x = self.drop_layer(x)
        
        # convert output shape for linear layer
        x = x.view(-1,16 * 56 * 56)
        
        # first linear layer
        x = F.relu(self.fc1(x))
        
        # output layers
        x1 = self.fc2(x) # team
        x2 = self.fc3(x) # player
        return x1, x2
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    return parser.parse_args()

def parse_bboxs(file):
    header = file.readline()
    assert header == 'tl_x,tl_y,br_x,br_y\n', header

    for line in file:
        line = tuple(int(round(float(i))) for i in line.split(','))
        tl = line[:2]
        br = line[2:]
        yield tl, br

def imcrop(img, bbox):
   x1, y1, x2, y2 = bbox
   if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
   return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    #args = parse_args()
    #Later to restore:
    model = Net()
    model.load_state_dict(torch.load( "team_and_person_model_2py.pt"))
    model.eval()
    
    team_colors = {0: (255,0,0),
                   1: (255, 0, 255),
                   2: (0, 0, 255),
                   3: (0, 255, 255),
                   4: (0, 255, 0),
                   5: (255, 255,0)}
    #images_directory = pathlib.Path(args.images)
    images_directory = pathlib.Path(r'C:\Users\dominicdathan\Documents\Personal\personal\player-team_assignment\part2')
    for image_path in images_directory.glob('*.jpg'):
        csv_path = image_path.with_suffix('.csv')
        assert csv_path.exists(), csv_path

        logging.info(f'displaying {image_path}')

        image = cv2.imread(str(image_path))
        assert image is not None, image_path
        
        tran = transforms.ToTensor()
        
        with open(csv_path, 'r') as csv_file:
            for tl, br in parse_bboxs(csv_file):
                
                height = br[1] - tl[1]
                width = br[0] - tl[0]
                crop = imcrop(image,[tl[0],tl[1],br[0],br[1]])
                crop_padded = resizeAndPad(crop, (224,224), 0)
                # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                # cv2.imshow('image', crop_padded)
                
                img_tensor = tran(crop_padded)
                img_tensor = img_tensor.unsqueeze(0)
                img_tensor.permute(0,3,2,1)
                team_pred_idx, person_pred_idx = model(img_tensor)
                team_id = team_pred_idx.argmax(1).item()
                team_color = team_colors[team_id]
                
                
                # cv2.waitKey(0)
                
                tile = cv2.rectangle(image, tl, br, team_color, 2)

        
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', round(image.shape[0]/2),round(image.shape[1]/2))
        cv2.imshow('image', image)

        if  cv2.waitKey(0) == ord('q'):
            logging.info('exiting...')
            exit()
