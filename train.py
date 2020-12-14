import torch


def train(loader, model, optimizer, criterion, device):
    model.train()
    corrects = 0
    for x,y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        optimizer.zero_grad()

        outputs = model(x)
        predicts = torch.argmax(outputs, 1).detach()
        corrects += (predicts == y).sum()

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return float(corrects)/float(len(loader.dataset))


def test(loader, model, device):
    model.eval()
    corrects = 0
    for x,y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        outputs = model(x)
        predicts = torch.argmax(outputs, 1).detach()
        corrects += (predicts == y).sum()

    return float(corrects)/float(len(loader.dataset))