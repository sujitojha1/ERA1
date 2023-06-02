
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # size: 28x28 > 26x26 | rf 1>3 | jump 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # size: 26x26 > 24x24 > 12x12 | rf 3>5>6 | jump 1>1>2
        x = F.relu(self.conv3(x), 2) #size 12x12 > 10x10 | rf 6 > 10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  # 10x10 > 8x8 > 4x4 | rf 10>14>16 | jump 2>2>4
        x = x.view(-1, 4096) # 4x4x256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def getModelSummary(model):

    from torchsummary import summary
    summary(model, input_size=(1, 28, 28))

from tqdm import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def training(model, device, num_epochs, train_loader, test_loader, optimizer, criterion):

    # Data to plot accuracy and loss graphs
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, criterion)
        test(model, device, test_loader, criterion)
        scheduler.step()

    return train_losses, test_losses, train_acc, test_acc, test_incorrect_pred

def getTrainingTestPlots(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")