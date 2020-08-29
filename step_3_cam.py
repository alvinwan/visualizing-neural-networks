"""Generate Class Activation Maps"""
import numpy as np
import sys
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.cm as cm

from PIL import Image
from step_2_pretrained import load_image


def load_raw_image():
    """Load raw 224x224 center crop of image"""
    image = Image.open(sys.argv[1])
    transform = transforms.Compose([
      transforms.Resize(224),  # resize smaller side of image to 224
      transforms.CenterCrop(224),  # take center 224x224 crop
    ])
    return transform(image)


def get_model():
    """Get model, set forward hook to save second-to-last layer's output"""
    net = models.resnet18(pretrained=True).eval()
    layer = net.layer4[1].conv2

    def store_feature_map(self, _, output):
        self._parameters['out'] = output
    layer.register_forward_hook(store_feature_map)

    return net, layer


def main():
    """Generate CAM for network's predicted class"""
    x = load_image()

    net, layer = get_model()

    out = net(x)
    _, (pred,) = torch.max(out, 1)  # get class with highest probability

    # 1. get second-to-last-layer output
    features = layer._parameters['out'][0]

    # 2. get weights w_1, w_2, ... w_n
    weights = net.fc._parameters['weight'][pred]

    # 3. compute weighted sum of output
    cam = (features.T * weights).sum(2)

    # normalize cam
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.detach().numpy()

    # save heatmap
    heatmap = (cm.jet_r(cam) * 255.0)[..., 2::-1].astype(np.uint8)
    heatmap = Image.fromarray(heatmap).resize((224, 224))
    heatmap.save('heatmap.jpg')
    print(' * Wrote heatmap to heatmap.jpg')

    # save heatmap on image
    image = load_raw_image()
    combined = (np.array(image) * 0.5 + np.array(heatmap) * 0.5).astype(np.uint8)
    Image.fromarray(combined).save('combined.jpg')
    print(' * Wrote heatmap on image to combined.jpg')



if __name__ == '__main__':
    main()
