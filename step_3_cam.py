from step_2_pretrained import load_image


def get_model():
    net = models.resnet18(pretrained=True).eval()


def main():
    x = load_image()
