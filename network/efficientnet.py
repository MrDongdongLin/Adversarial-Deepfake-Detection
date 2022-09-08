import io
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchsummary import summary


class MyEfficientNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        cnn_model_name = "efficientnet-b3"
        super(MyEfficientNet, self).__init__()
        if not pretrained:
            self.network = EfficientNet.from_name(cnn_model_name)
        else:
            self.network = EfficientNet.from_pretrained(cnn_model_name)

        self.network._fc = nn.Sequential(nn.Linear(self.network._fc.in_features, num_classes))

    def forward(self, images):
        out = self.network(images)
        return out


if __name__ == "__main__":
    # hyper parameters
    cnn_model_name = "efficientnet-b3"
    img_dim, img_height, img_width = 3, 224, 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyEfficientNet(1, True).to(device)

    # print(model)

    model_starts = summary(model, (3, img_height, img_width), depth=3)
    summary_str = str(model_starts)
    with io.open("../log/efficientnet_b3.txt", "w", encoding="utf-8") as f:
        f.write(summary_str)
