import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import time
import os
import warnings

warnings.filterwarnings("ignore")

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(25088, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=2):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = input.view(input.size(0), -1)
        input = self.fc(input)

        return input


class Model:
    def __init__(self, datapath, model, pretrained):
        self.datapath = datapath
        self.model = model
        self.pretrained = pretrained
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not os.path.isdir("checkpoints"):
            os.mkdir("checkpoints")

    def data_preparation(self):
        data_transform = {
            'train': transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])}

        train_dataset = datasets.ImageFolder(self.datapath, data_transform['train'])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

        return train_dataloader

    def train(self, net, train_iter, trainer, num_epochs):
        loss = nn.CrossEntropyLoss(reduction='sum')
        net.to(self.device)
        net.train()
        train_ep_loss = 0.0
        train_ep_acc = 0.0

        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, y in train_iter:
                X, y = X.to(self.device), y.to(self.device)
                trainer.zero_grad()
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                trainer.step()
                train_l_sum += l.item()
                train_acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
                n += y.shape[0]
            train_ep_loss += train_l_sum / n
            train_ep_acc += train_acc_sum / n

        train_loss = train_ep_loss / num_epochs
        train_acc = train_ep_acc / num_epochs

        return train_loss, train_acc

    def vgg16(self):
        net = VGG16()
        net.to(self.device)

        trainer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
        train_loss, train_acc = self.train(net, self.data_preparation(), trainer, 20)

        torch.save(net.state_dict(), 'checkpoints/vgg16.pth')

        return train_loss, train_acc

    def vgg16_pretrained(self):
        net = torchvision.models.vgg16(pretrained=True)
        net.to(self.device)

        for param in net.features.parameters():
            param.requires_grad = False

        net.classifier[6].out_features = 2
        trainer = torch.optim.SGD(net.classifier.parameters(), lr=0.0001, momentum=0.5)
        train_loss, train_acc = self.train(net, self.data_preparation(), trainer, 20)

        torch.save(net.state_dict(), 'checkpoints/vgg16_pretrained.pth')

        return train_loss, train_acc

    def resnet18(self):
        net = ResNet18(3, ResBlock, outputs=2)
        net.to(self.device)

        trainer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
        train_loss, train_acc = self.train(net, self.data_preparation(), trainer, 20)

        torch.save(net.state_dict(), 'checkpoints/resnet18.pth')

        return train_loss, train_acc

    def resnet18_pretrained(self):
        net = torchvision.models.resnet18(pretrained=True)
        net.to(self.device)

        for param in net.parameters():
            param.requires_grad = False

        net.fc = nn.Linear(in_features=512, out_features=2)

        params_to_update = []

        for name, param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        trainer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        train_loss, train_acc = self.train(net, self.data_preparation(), trainer, 20)

        torch.save(net.state_dict(), 'checkpoints/resnet18_pretrained.pth')

        return train_loss, train_acc

    def model_choice(self):
        print("In progress ...")

        if self.model == "VGG" and self.pretrained == 'False':
            train_loss, train_acc = self.vgg16()
        elif self.model == "VGG" and self.pretrained == 'True':
            train_loss, train_acc = self.vgg16_pretrained()
        elif self.model == "resnet18" and self.pretrained == 'False':
            train_loss, train_acc = self.resnet18()
        elif self.model == "resnet18" and self.pretrained == 'True':
            train_loss, train_acc = self.resnet18_pretrained()
        else:
            print("Something went wrong...")

        print("Train Loss: {}. Train accuracy: {}".format(train_loss, train_acc))


class ModelEval:
    def __init__(self, test_datapath, model, pretrained):
        self.test_datapath = test_datapath
        self.model = model
        self.pretrained = pretrained
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def data_preparation(self):
        data_transform = {
            'test': transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])}

        test_dataset = datasets.ImageFolder(self.test_datapath, data_transform['test'])
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

        return test_dataloader

    def model_choice(self):
        if self.model == "VGG" and self.pretrained == 'False':
            net = VGG16()
            path_state_dict = 'checkpoints/vgg16.pth'

        elif self.model == "VGG" and self.pretrained == 'True':
            net = torchvision.models.vgg16(pretrained=True)
            path_state_dict = 'checkpoints/vgg16_pretrained.pth'

        elif self.model == "resnet18" and self.pretrained == 'False':
            net = ResNet18(3, ResBlock, outputs=2)
            path_state_dict = 'checkpoints/resnet18.pth'

        elif self.model == "resnet18" and self.pretrained == 'True':
            net = torchvision.models.resnet18(pretrained=True)
            net.fc = nn.Linear(in_features=512, out_features=2)
            path_state_dict = 'checkpoints/resnet18_pretrained.pth'

        else:
            print("Something went wrong...")

        return net, path_state_dict

    def evaluate_accuracy(self, data_iter, net):
        acc_sum, n = torch.Tensor([0]), 0
        acc_sum = acc_sum.to(self.device)
        net.to(self.device)
        net.eval()
        for X, y in data_iter:
            X, y = X.to(self.device), y.to(self.device)
            acc_sum += (net(X).argmax(axis=1) == y).sum()
            n += y.shape[0]
        return acc_sum.item() / n

    def test_eval(self):

        print("In progress ...")

        net, path_state_dict = self.model_choice()
        net.to(self.device)
        net.load_state_dict(torch.load(path_state_dict))

        num_epochs = 5
        test_ep_acc = 0.0

        criterion = nn.CrossEntropyLoss(reduction='sum')
        test_loader = self.data_preparation()

        for epoch in range(num_epochs):

            test_ep_acc += self.evaluate_accuracy(test_loader, net)
            test_ep_loss = []

            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = net(X)
                loss = criterion(y_hat, y)
                test_ep_loss.append(loss.item())

            test_ep_loss = np.mean(test_ep_loss)

        test_acc = test_ep_acc / num_epochs
        test_loss = test_ep_loss / num_epochs

        print("Test loss: {}. Test accuracy: {}".format(test_loss, test_acc))


class ModelAugmentation:
    def __init__(self, datapath, model, pretrained):
        self.datapath = datapath
        self.model = model
        self.pretrained = pretrained
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not os.path.isdir("checkpoints"):
            os.mkdir("checkpoints")

    def data_preparation(self):
        data_transform = {
            'train': transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomResizedCrop((100, 100), scale=(0.1, 1), ratio=(0.5, 2)),
                transforms.ToTensor()
            ])}

        train_dataset = datasets.ImageFolder(self.datapath, data_transform['train'])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

        return train_dataloader

    def train(self, net, train_iter, trainer, num_epochs):
        loss = nn.CrossEntropyLoss(reduction='sum')
        net.to(self.device)
        net.train()
        train_ep_loss = 0.0
        train_ep_acc = 0.0

        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, y in train_iter:
                X, y = X.to(self.device), y.to(self.device)
                trainer.zero_grad()
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                trainer.step()
                train_l_sum += l.item()
                train_acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
                n += y.shape[0]
            train_ep_loss += train_l_sum / n
            train_ep_acc += train_acc_sum / n

        train_loss = train_ep_loss / num_epochs
        train_acc = train_ep_acc / num_epochs

        return train_loss, train_acc

    def vgg16_pretrained(self):
        net = torchvision.models.vgg16(pretrained=True)
        net.to(self.device)

        for param in net.features.parameters():
            param.requires_grad = False

        net.classifier[6].out_features = 2
        trainer = torch.optim.SGD(net.classifier.parameters(), lr=0.0001, momentum=0.5)
        train_loss, train_acc = self.train(net, self.data_preparation(), trainer, 20)

        torch.save(net.state_dict(), 'checkpoints/vgg16_pretrained.pth')

        return train_loss, train_acc

    def resnet18_pretrained(self):
        net = torchvision.models.resnet18(pretrained=True)
        net.to(self.device)

        for param in net.parameters():
            param.requires_grad = False

        net.fc = nn.Linear(in_features=512, out_features=2)

        params_to_update = []

        for name, param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        trainer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        train_loss, train_acc = self.train(net, self.data_preparation(), trainer, 20)

        torch.save(net.state_dict(), 'checkpoints/resnet18_pretrained.pth')

        return train_loss, train_acc

    def model_choice(self):
        print("In progress ...")

        if self.model == "VGG" and self.pretrained == 'True':
            train_loss, train_acc = self.vgg16_pretrained()
        elif self.model == "resnet18" and self.pretrained == 'True':
            train_loss, train_acc = self.resnet18_pretrained()
        else:
            print("Something went wrong...")

        print("Train Loss: {}. Train accuracy: {}".format(train_loss, train_acc))