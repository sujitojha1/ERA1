import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Train data transformations
# Apply a series of transformations to the training data
# These transformations include a random center crop, resizing, random rotation, and normalization of the data.
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),  # Randomly apply a centered crop to the image
    transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
    transforms.RandomRotation((-15., 15.), fill=0),  # Randomly rotate the image by a degree between -15 and 15
    transforms.ToTensor(),  # Convert the image to PyTorch tensor
    transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the tensor with mean and standard deviation
    ])

# Test data transformations
# Apply transformations to the test data.
# These transformations include converting the image to a tensor and normalizing the data.
test_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to PyTorch tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the tensor with mean and standard deviation
    ])

def getDataLoader(batch_size = 512):
    """
    Returns the train and test data loaders.

    This function loads the MNIST dataset from the specified directory, applies the transformations,
    and returns the DataLoader for both the train and test datasets.

    Args:
    batch_size (int, optional): The number of samples per batch. Defaults to 512.

    Returns:
    tuple: The DataLoader objects for the train and test datasets.
    """
   
    # Load the MNIST dataset for training, apply transformations and download if not present
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    
    # Load the MNIST dataset for testing, apply transformations and download if not present
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    # Create a DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    
    # Create a DataLoader for the train dataset
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    
    return train_loader, test_loader


def getSampleImages(data_loader):
    """
    Returns a plot of sample images from the dataset.

    This function takes in a DataLoader object and plots the first 12 images in a 3x4 grid.

    Args:
    data_loader (DataLoader): The DataLoader object to sample images from.

    Returns:
    matplotlib.figure.Figure: The Figure object with the plotted images.
    """
    # Get the first batch of images and labels from the DataLoader
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