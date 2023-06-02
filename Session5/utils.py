import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

def getDataLoader(batch_size = 512):
   
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)

    kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}

    test_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    
    return train_loader, test_loader


import matplotlib.pyplot as plt

def getSampleImages(data_loader):
    batch_data, batch_label = next(iter(data_loader)) 

    fig = plt.figure()

    for i in range(12):
      plt.subplot(3,4,i+1)
      plt.tight_layout()
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')
      plt.title(batch_label[i].item())
      plt.xticks([])
      plt.yticks([])
        
    return fig
