##########################################
# Stat231 Project 0:
# Deep Neural Network Classification on CIFAR-10
# Author: Feng Gao
##########################################

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

###################################################################################################
#                                              PART 1                                             #
###################################################################################################


def flatten(x):
    """
    Input:
    - Tensor of shape (N, D1, ..., DM)

    Output:
    - Tensor of shape (N, D1 * ... * DM)
    """

    ############################################################################
    # TODO: (1.a) Reshape tensor x into shape (N, D1 * ... * DM)               #
    ############################################################################
    x_flat = None
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
    return x_flat


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # TODO: Define your network structure
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        ############################################################################
        # TODO: (1.a), (2.a) Implement remaining forward pass.                     #
        ############################################################################

        self.block4 = None
        self.block5 = None

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x, viz=False):
        ############################################################################
        # TODO: (1.a), (2.a) Implement remaining forward pass.                     #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return flatten(x)


parser = argparse.ArgumentParser(description='stat231_projects')
parser.add_argument('--project', type=str, default=None)
############################################################################
# TODO: (1.b) Adjust learning-rate and number of epochs.                   #
############################################################################
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=5e+1)
############################################################################
#                             END OF YOUR CODE                             #
############################################################################
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--path', type=str, default='./results/model/')
parser.add_argument('--log', type=str, default='./results/log/')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load CIFAR-10 Dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

model = LeNet()
if args.cuda:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

if not os.path.exists(args.path):
    os.makedirs(args.path)
if not os.path.exists(args.log):
    os.makedirs(args.log)

train_loss = []
train_acc = []
test_loss = []
test_acc = []


def save_model(state, path):
    torch.save(state, os.path.join(path))


def train(epoch):
    model.train()
    training_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(trainloader):
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        _, predicted = output.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    train_loss.append(training_loss/len(trainloader))
    train_acc.append(100.*correct/total)
    print('Training Epoch:{}, Training Loss:{:.6f}, Acc:{:6f}'.format(epoch, training_loss/len(trainloader), 100.*correct/total))

def test(epoch):
    model.eval()
    testing_loss = 0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(testloader):
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        output = model(x)
        loss = criterion(output, y)
        
        testing_loss += loss.item()
        _, predicted = output.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    test_loss.append(testing_loss/len(testloader))
    test_acc.append(100.*correct/total)
    print('Testing Epoch:{}, Testing Loss:{:.6f}, Acc:{:6f}.'.format(epoch, testing_loss/len(testloader), 100.*correct/total))


def plot_loss(train_loss, test_loss):
    ############################################################################
    # TODO: (1.c) Plot.                                                        #
    ############################################################################
    pass


def plot_acc(train_acc, test_acc):
    ############################################################################
    # TODO: (1.c) Plot.                                                        #
    ############################################################################
    pass


def viz_cnn_filter(model):
    ############################################################################
    # TODO: (3.a) Visualize image of kernels.                                  #
    ############################################################################
    pass


def viz_response(model):
    ############################################################################
    # TODO: (3.b) Visualize filter response.                                   #
    ############################################################################
    pass

    
def main():
    epoch = 0
    while epoch < args.epochs:
        epoch += 1
        train(epoch)
        test(epoch)
    
    save_model({
        'network_params': model.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }, args.path+'model.pth')

    plot_loss(train_loss, test_loss)
    plot_acc(train_acc, test_acc)

    viz_cnn_filter(model)
    viz_response(model)


if __name__ == "__main__":
    main()
