__author__ = "Léopold Le Roux"
"""
Transfert learning script.
Will work with Mutilple pretrain CNN from pytorch. More information on : https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 
The experiments had work on ELO images. Those images had been croped to remove all borders.
The data have
Script create the 29/11/2019
Last revision: 02/12/2019

"""
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

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

@dataclass
class Project_data:
    """
    This class store the useful information of the project.
    Code inspired by: https://gist.github.com/FrancescoSaverioZuppichini/9711a48c4563980b438f40276d6db390
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'Data')
    checkpoint_dir = os.path.join(base_dir, 'Checkpoint')
    num_epochs = 100
    num_classes = 3
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = ["resnet", "alexnet", "vgg", "squeezenet", "densenet" ] # inception remove, crash, "inception"]
    feature_extract= True
    lr=0.001
    momentum=0.9

    def __post_init__(self):
        try:
            #create the directory if it don't exist
            os.mkdir(self.checkpoint_dir)
        except FileExistsError:
            #Delect the existing folder
            print("Pre existing folder have been found on ", self.checkpoint_dir)
            print("We will remove them and create new empty one")
            print("This was raised by the class Project_data __post__init")
            shutil.rmtree(self.checkpoint_dir, ignore_errors=False, onerror=None)
            os.mkdir(self.checkpoint_dir)        
        if(not os.path.isdir(self.data_dir)):
            msg = "No path data in " + str(self.data_dir)
            raise Exception(msg)

class ImgagesAugmentation:
    """
    Wrapper for image augmentation
    """
    def __init__(self):
        #image augmentation commands
        self.aug = T.Compose([
            T.RandomCrop(100),
            T.RandomHorizontalFlip(p=0.25),
            T.RandomVerticalFlip(p=0.25),
            T.Resize(224)
        ])
    def __call__(self, img):
        img = self.aug(img)
        return img

def get_dataloaders(train_dir, val_dir , train_transform= None , val_transform = None, split = (0.8 , 0.2), batch_size = 32):
    """
    Return train, val and test dataloader
    """
    train_ds  =  ImageFolder( root= train_dir, transform= train_transform)
    val_ds = ImageFolder( root= val_dir , transform= val_transform)
    #split the data in train and test
    lengths = np.array(split) * len(val_ds)
    lengths = lengths.astype(int)
    left = len(val_ds) - lengths.sum()
    lengths[-1] += left
    val_ds, test_ds = random_split(val_ds, lengths.tolist())
    print("Train samples :", len(train_ds), " , validation samples :", len(val_ds), " test samples :", len(test_ds))
    train_dl =  DataLoader(train_ds, batch_size= batch_size , shuffle= True)
    val_dl = DataLoader(val_ds, batch_size= batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return train_dl, val_dl, test_dl

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, use_pretrained = True,is_inception=False):
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

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def main (model_name, use_pretrained = True):
    """
    Create and train a model. The input is the name of the model.
    """

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, project.num_classes, project.feature_extract, use_pretrained= use_pretrained)
    # Print the model we just instantiated
    print(model_ft)
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(project.data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=project.batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
    # Send the model to GPU
    model_ft = model_ft.to(project.device)


    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if project.feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(params_to_update, project.lr, project.momentum)


    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=project.num_epochs,use_pretrained = use_pretrained ,is_inception=(model_name=="inception"))

if __name__ == "__main__":
    project = Project_data()
    for one_cnn in project.model_name:
        print("Working with ", one_cnn)
        main(model_name = one_cnn, use_pretrained= True)
    for one_cnn in project.model_name:
        print("Working with ", one_cnn)
        main(model_name = one_cnn, use_pretrained= False)
