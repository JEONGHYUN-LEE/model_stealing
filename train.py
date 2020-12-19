import torch
import numpy as np


def train_cam(loader, victim_model, attack_model, optimizer, criterion, lambda_, device):
    victim_model.eval()
    attack_model.train()
    corrects = 0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        optimizer.zero_grad()

        v_features_blobs = []

        def v_hook_feature(module, input, output):
            v_features_blobs.append(output)

        a_features_blobs = []

        def a_hook_feature(module, input, output):
            a_features_blobs.append(output)

        # a_handle = attack_model._modules.get('feature')._modules.get('4').register_forward_hook(a_hook_feature)
        # v_handle = victim_model._modules.get('feature')._modules.get('4').register_forward_hook(v_hook_feature)
        a_handle = attack_model.layer4[1].conv2.register_forward_hook(a_hook_feature)
        v_handle = victim_model.layer4[1].conv2.register_forward_hook(v_hook_feature)

        a_outputs = attack_model(x)
        a_predicts = torch.argmax(a_outputs, 1).detach()
        corrects += (a_predicts == y).sum()
        a_params = list(attack_model.parameters())
        a_weight_softmax = a_params[-2]
        a_featuremap = a_features_blobs[0]
        a_cams = torch.zeros(0).to(device)
        for i in range(len(a_featuremap)):
            a_cam = torch.tensordot(a_weight_softmax[a_predicts[i]],
                                    a_featuremap[i], dims=([0], [0]))
            a_cam = a_cam - torch.min(a_cam)
            a_cam = a_cam / torch.max(a_cam)
            a_cams = torch.cat((a_cams, a_cam.unsqueeze(0)), 0)

        with torch.no_grad():
            v_outputs = victim_model(x)
            v_predicts = torch.argmax(v_outputs, 1).detach()

            v_params = list(victim_model.parameters())
            v_weight_softmax = v_params[-2]
            v_featuremap = v_features_blobs[0]
            v_cams = torch.zeros(0).to(device)
            for i in range(len(v_featuremap)):
                v_cam = torch.tensordot(v_weight_softmax[v_predicts[i]],
                                        v_featuremap[i], dims=([0], [0]))
                v_cam = v_cam - torch.min(v_cam)
                v_cam = v_cam / torch.max(v_cam)
                v_cams = torch.cat((v_cams, v_cam.unsqueeze(0)), 0)

        cam_diff = (a_cams-v_cams).view(a_cams.shape[0], -1)
        # print(a_cams.max())
        # print(v_cams.max())
        # print(cam_diff.max())
        # loss = criterion(a_outputs, y)
        loss = criterion(a_outputs, y) + \
               lambda_ * (torch.norm(cam_diff, 'fro', 1) * torch.norm(cam_diff, 'fro', 1)).sum()
        # print(criterion(a_outputs, y).item(), (torch.norm(cam_diff, 'fro', 1) * torch.norm(cam_diff, 'fro', 1)).sum().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        a_handle.remove()
        v_handle.remove()
    return float(corrects)/float(len(loader.dataset))

# def train_cam(loader, victim_cam, attack_cam, victim_model, attack_model, optimizer, criterion, lambda_, device):
#     victim_model.eval()
#     attack_model.train()
#     corrects = 0
#     for x, y in loader:
#         x = x.to(device).float()
#         y = y.to(device).long()
#
#         optimizer.zero_grad()
#
#         outputs = attack_model(x)
#         predicts = torch.argmax(outputs, 1).detach()
#         corrects += (predicts == y).sum()
#         cam_diff = victim_cam(x, None)-attack_cam(x, None)
#         cam_diff = cam_diff.view(-1)
#         loss = criterion(outputs, y) + lambda_*torch.dot(cam_diff, cam_diff)
#         loss.backward()
#         optimizer.step()
#
#     return float(corrects)/float(len(loader.dataset))


def train(loader, model, optimizer, criterion, device):
    model.train()
    corrects = 0
    for x, y in loader:
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


def query(loader, model, device):
    model.eval()
    labels = np.zeros(0)
    for x, _ in loader:
        x = x.to(device).float()
        outputs = model(x)
        predicts = torch.argmax(outputs, 1).detach().cpu().numpy()
        labels = np.concatenate((labels, predicts), 0)
    return torch.tensor(labels)