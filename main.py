import matplotlib.pyplot as plt
import numpy as np

import torch                                # PyTorch package
import torchvision                          # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn                       # basic building block for neural neteorks
import torch.nn.functional as F             # import convolution functions like Relu
import torch.optim as optim                 # optimzer

import torch.utils.data

TRAIN_DATA_PATH = "./data/train/"
TEST_DATA_PATH = "./data/test/"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        ''' Models a simple Convolutional Neural Network'''

        def __init__(self):
            ''' initialize the network '''
            super(Net, self).__init__()
            # 3 input image channel, 6 output channels,
            # 5x5 square convolution kernel
            self.conv1 = nn.Conv2d(3, 6, 5)
            # Max pooling over a (2, 2) window
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 8, 5)
            self.fc1 = nn.Linear(8 * 5 * 5, 120)  # 5x5 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)
            # self.gfy1 = nn.Linear(120, fc3[1]*num_batches/100)

        def forward(self, x):
            ''' the forward propagation algorithm '''
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 8 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net().to(device)

    scaler = 0.8
    transform = transforms.Compose([
        transforms.Resize((int(1328 * scaler), int(1160 * scaler))),  # from 1328x1160
        transforms.ToTensor(),  # to tensor object
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # mean = 0.5, std = 0.5
    ])

    # set batch_size
    batch_size = 4

    # set number of workers
    num_workers = 2

    # load train data
    trainset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
    ## append also transformed data to artificially generate more input
    ## TODO: search for how to append
    # trainset.append( torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH,
    #                               transform=transform,
    #                               target_transform=transforms.GaussianBlur) )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    # load test data
    testset = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    # put 1 class into a set
    classes = 'pneumonia'

    def imshow(img):
        ''' function to show image '''
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()  # convert to numpy objects

        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get random training images with iter function
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))

    # print the class of the image
    # print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()  # loss of entire mini-batch
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # save
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    # reload
    net = Net()
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            print(labels.size(0))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == '__main__':
    main()



