from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import cv2

features = []


def append_final_conv(module, input, output):
    features.append(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def process_name(fname):
    fname = fname.split('.')
    extention = fname[-1]
    return '.'.join(fname[:-1]) + '_CAM.' + extention


def save(model, layer_name, image_fname, cam_location, display=False):
    img = cv2.imread(image_fname)
    height, width, _ = img.shape

    model.eval()

    # features_blobs = []
    model._modules.get(layer_name).register_forward_hook(append_final_conv)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  # FIXME: Do we want to change normalize to fit our dataset?
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])  # TODO: Also, did we use 224 by 224?

    img_tensor = preprocess(Image.open(image_fname))
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = model(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    CAMs = returnCAM(features[-1], weight_softmax, [idx[0]])

    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)),
                                cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5

    cv2.imwrite(str(Path(cam_location)/process_name(image_fname)), result)

    if display:
        img = plt.imread(image_fname)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        result = heatmap * 0.3 + img * 0.5
        plt.imshow(result.astype(np.int32))
        plt.show()

if __name__ == "__main__":
    model_id = 1
    if model_id == 1:
        net = models.squeezenet1_1(pretrained=True)
        finalconv_name = 'features'
    elif model_id == 2:
        net = models.resnet18(pretrained=True)
        finalconv_name = 'layer4'
    elif model_id == 3:
        net = models.densenet161(pretrained=True)
        finalconv_name = 'features'

    save(net, finalconv_name, 'test.jpg', '.', True)
