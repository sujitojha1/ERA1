import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


class Net(nn.Module):
    """A Convolutional Neural Network (CNN) model with 4 convolutional layers and 2 linear layers.
    
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 1 input channel, 32 output channels, and a kernel size of 3.
        conv2 (nn.Conv2d): Second convolutional layer with 32 input channels, 64 output channels, and a kernel size of 3.
        conv3 (nn.Conv2d): Third convolutional layer with 64 input channels, 128 output channels, and a kernel size of 3.
        conv4 (nn.Conv2d): Fourth convolutional layer with 128 input channels, 256 output channels, and a kernel size of 3.
        fc1 (nn.Linear): First linear layer with 4096 input features and 50 output features.
        fc2 (nn.Linear): Second linear layer with 50 input features and 10 output features.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the layers and activation functions.
        """
        # Convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))

        # Flatten the tensor
        x = x.view(-1, 4096)

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

def getModelSummary(model):
    """Prints the summary of the model.

    Args:
        model (Net): The model to print the summary of.
    """
    from torchsummary import summary
    summary(model, input_size=(1, 28, 28))

def GetCorrectPredCount(pPrediction, pLabels):
    """Returns the count of correct predictions.

    Args:
        pPrediction (Tensor): The predicted labels.
        pLabels (Tensor): The ground truth labels.

    Returns:
        int: The count of correct predictions.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
    """Trains the model and updates the model parameters.

    Args:
        model (Net): The model to train.
        device (Device): The device to run the training on.
        train_loader (DataLoader): The data loader for the training data.
        optimizer (Optimizer): The optimization algorithm.
       ```python
        criterion (Loss): The loss function.
        train_losses (list): A list to store the loss value for each epoch.
        train_acc (list): A list to store the accuracy for each epoch.
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # Move data and target to device
        data, target = data.to(device), target.to(device)

        # Reset the gradients to zero
        optimizer.zero_grad()

        # Forward pass
        pred = model(data)

        # Compute loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update correct and processed count
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        # Update progress bar
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    # Append loss and accuracy for this epoch
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion, test_losses, test_acc):
    """Tests the model and updates the test loss and accuracy.

    Args:
        model (Net): The model to test.
        device (Device): The device to run the testing on.
        test_loader (DataLoader): The data loader for the test data.
        criterion (Loss): The loss function.
        test_losses (list): A list to store the loss value for each test.
        test_acc (list): A list to store the accuracy for each test.
    """
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Move data and target to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Compute loss
            test_loss += criterion(output, target).item()  # sum up batch loss

            # Update correct count
            correct += GetCorrectPredCount(output, target)

    # Compute average loss and accuracy
    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def training(model, device, num_epochs, train_loader, test_loader, optimizer, criterion):
    """Trains and tests the model for a certain number of epochs.

    Args:
        model (Net): The model to train and test.
        device (Device): The device to run the training and testing on.
        num_epochs (int): The number of epochs.
        train_loader (DataLoader): The data loader for the training data.
        test_loader (DataLoader): The data loader for the test data.
        optimizer (Optimizer): The optimization algorithm.
        criterion (Loss): The loss function.

    Returns:
        tuple: The training losses, test losses, training accuracy, and test accuracy.
    """
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, criterion, train_losses, train_acc)
        test(model, device, test_loader, criterion, test_losses, test_acc)

    return train_losses, test_losses, train_acc, test_acc, test_incorrect_pred

def getTrainingTestPlots(train_losses, test_losses, train_acc, test_acc):
    """Generates and displays plots for training and test losses and accuracy.

    Args:
        train_losses (list): A list of training losses.
        test_losses (list): A list of test losses.
        train_acc (list): A list of training accuracy values.
        test_acc (list): A list of test accuracy values.
    """
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
