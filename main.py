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
        self.fc1 = nn.Linear(28 * 28, 10)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data = data.view(-1, 40)
        train_loss = 0.0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        if batch_idx % args.log_interval == 0:
            print(data.size(0))
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                # epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()  # prep model for *evaluation*
    for data, target in test_loader:
        # data = data.view(-1, 40)
        output = model(data)
        # sum up batch loss
        loss = F.nll_loss(output, target)
        test_loss += loss.item()*data.size(0)
        pred = torch.max(output, 1)  # convert output probabilities to predicted class
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))  # compare predictions to true label
        # calculate the test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

# calculate and print the average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    # return 100. * np.sum(class_correct) / np.sum(class_total)


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
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving ther current Model')
    parser.add_argument('--sliding-size', type=int, default=3, metavar='N',
                        help='sliding window size (one hand) for pre-processing data')
    parser.add_argument('--normalization', type=str, default='all', metavar='N',
                        help='how to do data normalization')
    parser.add_argument('--test-model', type=str, default='', metavar='N',
                        help='If test-model has a name, do not do training, just testing on dev and train set')
    parser.add_argument('--load-model', type=str, default='', metavar='N',
                        help='If load-model has a name, use pretrained model')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='N',
                        help='which optimizer to use')
    parser.add_argument('--split', type=int, default=3, metavar='N',
                        help='number of splits for training data')

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
                                               num_workers=num_workers)
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
        for samll_epoch in range(args.split):
            correct_rate_train.append(test(args, model, device, train_loader))
        print("Accurate rate on training set: " + str(np.array(correct_rate_train).mean()))
        return

    timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    os.mkdir(timeStr + "model")
    if len(args.load_model) > 1:
        model.load_state_dict(torch.load(args.load_model))
    generate(args, model, device, test_loader, timeStr+"model", 0)

    for epoch in range(1, args.epochs + 1):
        for samll_epoch in range(args.split):
            train(args, model, device, train_loader, optimizer, epoch)
            train_loader = None
        correct_rate = test(args, model, device, test_loader)

        if args.save_model:
            torch.save(model.state_dict(), timeStr+"model/"+str(epoch)+":"+str(correct_rate)+".pt")
            generate(args, model, device, test_loader, timeStr + "model", epoch)


if __name__ == '__main__':
    main()

