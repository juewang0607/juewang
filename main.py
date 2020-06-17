import argparse
import csv
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
import argparse
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import time
import os
batch_size = 20  # the numbers of samples per batch to load
num_workers = 0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.act1 = nn.ReLU()
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(10, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.act2 = nn.ReLU()
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)
        self.act3 = nn.ReLU()
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    # a four to five MLP with around 100 could be a good start point, for model architecture you may search around the internet
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation functio
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)

        #  x = self.fc3(x)  # just a remind that we do not need to add relu after final layer

        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    # number of epochs to train the model
    #n_epochs = 30  # suggest training between 20-50 epochs
    model.train()  # prep model for training
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        # monitor training loss
        train_loss = 0.0
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)
        # print training statistics
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss/len(train_loader)))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data = data.view(-1, 40)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return 100. * correct / len(test_loader.dataset)


def generate(args, model, device, test_loader, file_name, epoch):
    model.eval()
    f = open(file_name+"/"+str(epoch)+".csv", 'w')
    write = csv.writer(f)
    write.writerow(['id', 'label'])
    id_number = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data = data.view(-1, 40)
            output = model(data)
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            for i in range(pred.shape[0]):
                write.writerow([str(id_number + i), str(int(pred[i]))])
            id_number += pred.shape[0]


def main():
    # Training settings
    # clean up useless parameter here
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',  # adam's default learning rate is 0.001, 0.1 is too large
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='N',
                        help='which optimizer to use')
    parser.add_argument('--test-model', type=str, default='', metavar='N',
                        help='If test-model has a name, do not do training, just testing on dev and train set')
    parser.add_argument('--load-model', type=str, default='', metavar='N',
                        help='If load-model has a name, use pretrained model')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving ther current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print("Using device: "+str(device))

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='data', train=True,
                                download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False,
                               download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers) # check whether data is shuffled here
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    model = Net()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    if len(args.test_model) > 1:
        model.load_state_dict(torch.load(args.test_model))
        model.eval()
        correct_rate_train = []
        #for samll_epoch in range(args.split):
        correct_rate_train.append(test(args, model, device, train_loader))
        print("Accurate rate on training set: " + str(np.array(correct_rate_train).mean()))
        return

    timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    os.mkdir(timeStr + "model")
    if len(args.load_model) > 1:
        model.load_state_dict(torch.load(args.load_model))
    generate(args, model, device, test_loader, timeStr+"model", 0)

    for epoch in range(1, args.epochs + 1):
        #for samll_epoch in range(args.split):
        train(args, model, device, train_loader, optimizer, epoch)
            # train_loader = None
        correct_rate = test(args, model, device, test_loader)
        if args.save_model:
            torch.save(model.state_dict(), timeStr+"model/"+str(epoch)+":"+str(correct_rate)+".pt")
            generate(args, model, device, test_loader, timeStr + "model", epoch)


if __name__ == '__main__':
    main()
