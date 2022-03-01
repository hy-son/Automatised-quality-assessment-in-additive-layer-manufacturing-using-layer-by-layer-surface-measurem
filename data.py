
from dataclasses import dataclass
import os
import torch
import shutil
from torchvision import transforms

@dataclass
class Project_data:
    """
    This class store the useful information of the project.
    Code inspired by: https://gist.github.com/FrancescoSaverioZuppichini/9711a48c4563980b438f40276d6db390
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    #data_dir = os.path.join(base_dir, 'data')
    data_dir = r"."
    data_dir = r"Data"
    checkpoint_dir = os.path.join(base_dir, 'Checkpoint')
    trainned_dir =  [#os.path.join(checkpoint_dir, "SqueezeNet_pretrain_epoch-46.pt"),
                    #r"NeuralNet\ResNet_pretrain_epoch-10.pt",
                    r"NeuralNet\ResNet_pretrain_epoch-40.pt",
                    r"NeuralNet\SqueezeNet_pretrain_epoch-38.pt",
                    r"NeuralNet\DenseNet_pretrain_epoch-40.pt",
                    ]
    num_epochs = 100
    num_classes = 3
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = ["resnet", "squeezenet", "densenet" ]  #["resnet", "alexnet", "vgg", "squeezenet", "densenet" ] # inception remove, crash, "inception"]
    feature_extract= False
    lr=0.001
    momentum=0.9

def data_transforms(input_size):
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

    return data_transforms

