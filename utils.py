import torch
import torchvision
from torchvision.transforms import transforms
import models


def get_models(model_name):
    if model_name == 'resnet20':
        return models.resnet20()
    elif model_name == 'modelA':
        return models.ModelA()


def get_dataloader(dataset_name):
    trainloader, testloader = None, None
    if dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
                transforms.ToTensor()
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                 shuffle=False, num_workers=2)

    return trainloader, testloader