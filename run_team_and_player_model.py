""" 
run_team_and_person_model

script to load saved model and run on directory of image tiles given as argument

"""

# imports
import numpy as np
import os
import random
import json

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

# imports for parsing aurguements and path library
import argparse
import pathlib

class PlayerTilesDataset_Predictor(Dataset):
    """Image Loader dataset for player tiles."""
    def __init__(self, dirname, training_labels, transform=None):
        """ init method
        Args:
            dirname (string): Path to the images organized in a team\player\image.jpg format.
            training_labels (dict): dictionary mapping team and person labels to index
            transform: Any Pytorch transform to be applied
        """
        
        # initalise image paths
        self.image_paths = []
        
        # initialse image labels - one for team and one for player
        self.labels1 = []
        self.labels2 = []
        
        # get all team folders
        teams = [f.name for f in os.scandir(dirname) if f.is_dir()]

        # iterate over each team folder
        for team in list(teams):

            # create team directory
            teams_dir = os.path.join(dirname,team)

            # get all players for that team
            players = [f.name for f in os.scandir(teams_dir) if f.is_dir()]

            # iterate over each player
            for player in list(players):

                # create player directory
                player_dir = os.path.join(dirname,team,player)

                # get all images for that player
                images = [f.name for f in os.scandir(player_dir)]

                # iterate over all images
                for image in images:

                    # append image path
                    self.image_paths.append(os.path.join(dirname,team,player,image))
                    
                    # append labels
                    self.labels1.append(team)
                    self.labels2.append(team+'_'+player)

        
        # Create a dictionary mapping each label to a index from 0 to len(classes) for both outputs
        self.label1_to_idx = {x:training_labels['Team'][x] for x in self.labels1} # team
        self.label2_to_idx = {x:training_labels['Person'][x] for x in self.labels2} # player
        
        # transform if requessted
        self.transform = transform
        
    def __len__(self):
        # return length of dataset
        return len(self.image_paths)
      
    def __getitem__(self, idx):
        """ getitem method
        open one image, transform if required and and send along with corresponding labels
        
        Args:
            idx: index of image within dataset to get

        """
        
        # get image path
        img_name = self.image_paths[idx]
        
        # get image labels
        label1 = self.labels1[idx] # team
        label2 = self.labels2[idx] # player
        
        # open image
        image = Image.open(img_name)
        
        # transform if requested
        if self.transform:
            image = self.transform(image)
            
        return image,self.label1_to_idx[label1],self.label2_to_idx[label2]
    
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

# function to return the correct number of predictions from predictions and labels
def get_num_correct(preds, labels):
    
    corr1 = preds[0].argmax(dim=1).eq(labels[0])
    corr2 = preds[1].argmax(dim=1).eq(labels[1])
    
    corr = torch.logical_and(corr1,corr2)
    
    return corr.sum().item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    
    # get directory of folder holding all images from input arguement
    # nb. path to the images must be organized in a team\player\image.jpg format
    imdir = pathlib.Path(args.image_folder)
        
    # load model
    model = Net()
    print("Loading team_and_person_model_python.pt")
    model.load_state_dict(torch.load( "team_and_person_model_python.pt"))
    # set to evaluation state
    model.eval()
    
    # load model labels
    print("Loading team_and_person_model_labels.json")
    with open("team_and_person_model_labels.json") as json_file:
        train_labels = json.load(json_file)
    
    # transform images to tensor
    t = transforms.ToTensor()
    
    # load dataset
    print("Fetching images")
    test_dataset = PlayerTilesDataset_Predictor(dirname=imdir,training_labels=train_labels, transform=t)
    
    # calculate number of images
    num_image_tiles = len(test_dataset)
    print('{} images found to predict'.format(num_image_tiles))
    
    # create dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=500,shuffle=False)
    
    test_predictions = {'Team':[], 'Person':[]}
    test_targets = {'Team':[], 'Person':[]}
    test_total_correct = 0
    
    print("Processing images...")
    # iterate over each test batch
    for batch in test_loader: # get batch
        test_images, test_labels1, test_labels2 = batch 
        test_preds1, test_preds2 = model(test_images) # get predictions for batch
        test_predictions['Team'].extend(test_preds1.argmax(dim=1)) # append predictions for team
        test_predictions['Person'].extend(test_preds2.argmax(dim=1)) # append predictions for person
        test_targets['Team'].extend(test_labels1) # append team truth
        test_targets['Person'].extend(test_labels2) # append person truth
        # get total correct
        test_total_correct += get_num_correct([test_preds1,test_preds2], 
                                               [test_labels1,test_labels2]) 
        
    
    print("Completed predictions")
    print('Total correct: {} out of {} = {:.1f}%'.format(test_total_correct,num_image_tiles,100*test_total_correct/num_image_tiles))
