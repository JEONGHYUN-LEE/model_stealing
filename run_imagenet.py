import os
import yaml
import numpy as np
from torch.utils import data
import torch.optim as optim
from munch import munchify
from tqdm import tqdm
from utils import *
from train import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from grad_cam import *
from torchvision import datasets, transforms as T
import torchvision
import random


torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def run(config):

    params = munchify(config)
    device = params.device

    # Load Victim Model
    victim_model = torchvision.models.resnet18(pretrained=True, progress=True)
    victim_model.to(device)

    # Load Victim Model
    attack_model = torchvision.models.resnet18(pretrained=False, progress=True)
    attack_model.to(device)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load Full Dataset
    train_dataset = datasets.ImageNet("./data/10_class_imagenet/", split="train", transform=transform)
    seed_dataset = datasets.ImageNet("./data/10_class_imagenet/", split="val", transform=transform)
    test_dataset = datasets.ImageNet("./data/10_class_imagenet/", split="val", transform=transform)

    # Seed Trainset Indice
    # indices = list(range(len(seed_dataset)))
    # rng = np.random.RandomState(1)
    # rng.shuffle(indices)
    # seed_idx = indices[:params.seed_size]
    # print(seed_idx)
    # seed_sampler = SubsetRandomSampler(seed_idx)
    # Get Seed Dataset (Subset of Full Testset)
    seed_loader = torch.utils.data.DataLoader(seed_dataset,
                                              batch_size=params.seed_size,
                                              # sampler=seed_sampler,
                                              shuffle=True)
    data_iter = iter(seed_loader)
    X, y = data_iter.next()
    print(y)
    # Testset Loader
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=params.batch_size,
                                              shuffle=False)

    # Optimization Setup
    optimizer = optim.SGD(attack_model.parameters(), lr=params.lr, momentum=0.9, weight_decay=params.decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20,30])
    criterion = nn.CrossEntropyLoss()

    # # Init GramCAM
    # victim_cam = GradCam(model=victim_model, feature_module=victim_model.feature,
    #                      target_layer_names=["2"], device='cuda:7')
    # attack_cam = GradCam(model=attack_model, feature_module=attack_model.feature,
    #                      target_layer_names=["2"], device='cuda:7')

    train_accs = []
    test_accs = []
    minus = 1
    for aug_iter in tqdm(range(params.aug_iters)):
        if aug_iter % 2 == 0 and aug_iter != 0:
            minus *= -1
        # Update Train Loader
        aug_dataset = data.TensorDataset(X, y)
        print(minus)
        print(len(X))
        train_loader = torch.utils.data.DataLoader(aug_dataset,
                                                   batch_size=params.batch_size,
                                                   shuffle=False)
        for epoch in range(params.epochs):
            if params.cam:
                train_acc = train_cam(train_loader, victim_model,
                                      attack_model, optimizer, criterion, params._lambda, device)
            else:
                train_acc = train(train_loader, attack_model, optimizer, criterion, device)

            test_acc = test(test_loader, attack_model, device)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            #         scheduler.step()
            print("epoch: ", epoch,
                  "train_acc: ", round(train_acc, 4),
                  "test_acc: ", round(test_acc, 4))

        if params.augment and not (aug_iter == params.aug_iters-1):
            print("augment ..")
            # Augmented Dataset (X only)
            X_new = jacobian_augmentation(attack_model, X, y, device, 0.1, minus)

            # Query for Label of Augmented Dataset
            y_dummy = torch.zeros(len(X_new))
            query_dataset = data.TensorDataset(X_new, y_dummy)
            query_loader = torch.utils.data.DataLoader(query_dataset,
                                                       batch_size=params.batch_size,
                                                       shuffle=False)
            y_new = query(query_loader, victim_model, device)

            # Concatenate Augmented Dataset
            X = torch.cat([X, X_new[y_new < 10]], 0)
            y = torch.cat([y, y_new[y_new < 10].type(torch.LongTensor)], 0)
            # torch.save([X, y], 'augment_sample.pt')

    path, file = os.path.split(params.save_path)
    os.makedirs(path, exist_ok=True)
    torch.save({'train_accs': train_accs, 'test_accs': test_accs, 'model_param': attack_model.state_dict()},
               params.save_path)


def jacobian(model, x, device, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x = torch.unsqueeze(x, 0)
    x = x.to(device)
    # derivatives for each class
    for class_ind in range(nb_classes):
        x.requires_grad = True
        model.zero_grad()
        score = model(x)[:, class_ind]
        model.zero_grad()
        score.backward()
        list_derivatives.append(x.grad.data.cpu())
    return list_derivatives


def jacobian_augmentation(model, X, y, device, _lambda=0.1, minus=1):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_new = []

    # For each input in the previous' substitute training iteration
    for ind, (data, label) in enumerate(zip(X, y)):
        # Get jacobian matrix
        grads = jacobian(model, data, device)
        # Select gradient corresponding to the label predicted by the oracle
        # print(len(grads))
        # print(label)
        grad = grads[label]
        # Compute sign matrix
        grad_val = torch.sign(grad)

        # Create new synthetic point in adversary substitute training set
        if minus == -1:
            X_new.append(torch.clamp(X[ind] - _lambda * grad_val, 0.0, 1.0))
        else:
            X_new.append(torch.clamp(X[ind] + _lambda * grad_val, 0.0, 1.0))

    # Return new X
    X_new = torch.cat(X_new)
    return X_new
