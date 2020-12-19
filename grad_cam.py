import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            #             elif "avgpool" in name.lower():
            #                 print(module, x.shape)
            #                 x = module(x)
            #                 x = x.view(x.size(0),-1)
            #                 print(module, x.shape)
            else:
                x = x.view(x.size(0), -1)
                x = module(x)

        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, device):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.device = device
        self.model = model.to(device)

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        features, output = self.extractor(input.to(self.device))

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), 1)
        #             print(index.shape)

        one_hot = np.zeros((len(index), output.size()[-1]), dtype=np.float32)
        #         print(index)
        one_hot[torch.arange(len(index)), [index]] = 1
        #         print(one_hot)
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        #         print(output.shape, one_hot.shape)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu()

        target = features[-1]
        # print(target.shape)
        target = target.cpu()
        # print(target.shape)
        # print(target.shape, grads_val.shape)
        weights = torch.mean(grads_val, dim=(2, 3))
        cam = torch.zeros((target.shape[0], target.shape[2], target.shape[3]), dtype=torch.float32)
        # print("w", weights.shape, cam.shape)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                #             w (100, 64) cam (100, 20, 20) target 100 64 20 20
                cam[i, :, :] += weights[i, j] * target[i, j, :, :]

        cam = torch.nn.functional.relu(cam)
        # resized_cam = []
        # for i in range(len(cam)):
        #     resized_cam.append(cv2.resize(cam[i], input.shape[2:]))
        # resized_cam = torch.tensor(resized_cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam.to(self.device)

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

# if __name__ == '__main__':
#     """ python grad_cam.py <path_to_image>
#     1. Loads an image with opencv.
#     2. Preprocesses it for VGG19 and converts to a pytorch variable.
#     3. Makes a forward pass to find the category index with the highest score,
#     and computes intermediate activations.
#     Makes the visualization. """

#     args = get_args()

#     # Can work with any model, but it assumes that the model has a
#     # feature method, and a classifier method,
#     # as in the VGG models in torchvision.
#     model = models.resnet50(pretrained=True)
#     grad_cam = GradCam(model=model, feature_module=model.layer4, \
#                        target_layer_names=["2"], use_cuda=args.use_cuda)

#     img = cv2.imread(args.image_path, 1)
#     img = np.float32(cv2.resize(img, (224, 224))) / 255
#     input = preprocess_image(img)

#     # If None, returns the map for the highest scoring category.
#     # Otherwise, targets the requested index.
#     target_index = None
#     mask = grad_cam(input, target_index)

#     show_cam_on_image(img, mask)

# #     gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
# #     print(model._modules.items())
# #     gb = gb_model(input, index=target_index)
# #     gb = gb.transpose((1, 2, 0))
# #     cam_mask = cv2.merge([mask, mask, mask])
# #     cam_gb = deprocess_image(cam_mask*gb)
# #     gb = deprocess_image(gb)

# #     cv2.imwrite('gb.jpg', gb)
# #     cv2.imwrite('cam_gb.jpg', cam_gb)
