import os
import yaml
import torch.nn as nn
import torch.optim as optim
from munch import munchify
from tqdm import tqdm
from utils import *
from train import *


def run(config_path):
    # Load Experiment Config
    with open(config_path, "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        print(params)
    params = munchify(params)

    # Setup Training
    device = params.device
    model = get_models(params.network)
    model = model.to(device)
    train_loader, test_loader = get_dataloader(params.dataset)
    optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, params.lr_steps)
    criterion = nn.CrossEntropyLoss()

    # Run Train-Test Iteration
    train_accs = []
    test_accs = []
    best_test_acc = 0
    for epoch in tqdm(range(params.epochs)):
        # Train-Test
        train_acc = train(train_loader, model, optimizer, criterion, device)
        test_acc = test(test_loader, model, device)
        # Save If Best
        if best_test_acc < test_acc:
            path, file = os.path.split(params.save_path)
            os.makedirs(path, exist_ok=True)
            torch.save(model.to('cpu').state_dict(), params.save_path)
            model.to(device)
            best_test_acc = test_acc
        # Record Learning Curve
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        scheduler.step()
        # Print Result
        print("epoch: ", epoch, "train_acc: ", round(train_acc, 4), "test_acc: ", round(test_acc, 4))
    # Return Learning Curve and Best Result
    return train_accs, test_accs, best_test_acc