from dataclasses import dataclass
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



def train_model(model, dataloaders, criterion, optimizer, project , num_epochs=25,  use_pretrained = True,is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #Monitoring with tensorboard
    logdir =  str(model.__class__.__name__) + "_Pretrain_"+ str(use_pretrained) + "_" + str(time.time())
    writer =  SummaryWriter(log_dir=os.path.join(project.checkpoint_dir, logdir), comment= str(model.__class__.__name__))
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(project.device)
                labels = labels.to(project.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        #Monitoring with tensorboard
        writer.add_scalar('Train Loss', epoch_loss , epoch)
        writer.add_scalar('Train Accuracy', epoch_acc , epoch)
        writer.add_scalar('Test Loss', epoch_loss , epoch)
        writer.add_scalar('Test Accuracy', epoch_acc , epoch)
        writer.close()
        print("Monitored with tensorboard")

        #Save the model
        save_path = os.path.join(str(model.__class__.__name__), os.path.join(project.checkpoint_dir, '{}_epoch-{}.pt'.format(model.__class__.__name__,epoch)))
        torch.save(model.state_dict(), save_path  )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history