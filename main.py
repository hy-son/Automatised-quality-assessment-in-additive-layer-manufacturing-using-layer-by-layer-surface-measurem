__author__ = "Léopold Le Roux"
"""
Reprise de 5CNN to classify the images
Script create the 31/01/2020
Last revision: 09/02/2020

"""
#from dataclasses import dataclass
import os
import shutil
import numpy as np
import torchvision.transforms as T
import torch
import torchvision
from torchvision import transforms

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.folder import ImageFolder
from torchvision import models
import time
from torch.utils.tensorboard import SummaryWriter
import copy

from data import Project_data
from data import data_transforms
from model_creation import initialize_model
#from trainning import train_model
from evaluation import evaluation

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)



def main (model_name,project  ,path_trainned_CNN, use_pretrained = True):
   
    model_ft, input_size = initialize_model(model_name, project.num_classes, project.feature_extract, use_pretrained= use_pretrained)
    transforms = data_transforms(input_size)
    print("Initializing Datasets and Dataloaders...")
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(project.data_dir, x), transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=project.batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

    model_ft.load_state_dict( torch.load(path_trainned_CNN, map_location = project.device))

    running_corrects , results, accuracy = evaluation( model = model_ft, dataloaders = dataloaders_dict , project = project)

    return running_corrects , results, accuracy


if __name__ == "__main__":
    project = Project_data
    display = []
    for (cnn , path_cnn) in zip(project.model_name, project.trainned_dir):
        print("Working with ", cnn)
        running_corrects , results, accuracy = main(model_name = cnn, project = project, path_trainned_CNN = path_cnn ,use_pretrained= True)
        print( "******************************")
        print(cnn , " have an accuracy of ", accuracy)
        display.append([cnn,accuracy ])

    for i in range( len(display)):
        print( str(display[i][0]) , " | ",str(display[i][1])  )


    