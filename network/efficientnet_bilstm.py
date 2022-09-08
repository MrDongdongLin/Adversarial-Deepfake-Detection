import io
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchsummary import summary


class TruncEfficientLSTM(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True):
        super(TruncEfficientLSTM, self).__init__()

        # efficientnetb3
        cnn_last_size = 1536
        self.hidden_size = 300
        if pretrained:
            self.cnn_encoder = EfficientNet.from_pretrained(model_name)
        else:
            self.cnn_encoder = EfficientNet.from_name(model_name)

        self.lstm = nn.LSTM(
            cnn_last_size, self.hidden_size, num_layers=3, bias=True,
            batch_first=True, bidirectional=True, dropout=0.5
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # for bidirectional, hidden_size * 2
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, images):
        batch_size, time, num_channels, height, width = images.shape
        # (h, c) = (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))
        hidden = None  # default to zero, the same as above
        embed = []
        for t in range(time):
            inp = images[:, t, :, :, :].squeeze(1)  # (-1, 3, t, 300, 300) -> (-1, 3, 300, 300)
            x = self.cnn_encoder.extract_features(inp)  # (-1, 3, 300, 300) -> (-1, 1536, 7, 7)
            x = self.avg_pooling(x)  # (-1, 1536, 7, 7) -> (-1, 1536, 1, 1)
            x = x.flatten(start_dim=1)  # (-1, 1536, 1, 1) -> (-1, 1536)
            embed.append(x)  # (time, -1, 1536)
        embed = torch.stack(embed, dim=0)
        # we need to permute the time dimension since batch_first=True will set batch as the 1st dimension
        embed = embed.permute(1, 0, 2)
        out, _ = self.lstm(embed, hidden)  # (-1, time, 1536) -> (-1, time, hidden_size*direction)

        out = self.avg_pool(out.permute(0, 2, 1)).squeeze(2)  # (-1, time, 300*2) -> (-1, 600)
        x = self.fc(out)  # (-1, 600) -> (-1, num_classes)
        return x


class TruncEfficientLSTMV2(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True):
        super(TruncEfficientLSTMV2, self).__init__()

        # efficientnetb3
        cnn_last_size = 1536
        self.hidden_size = 300
        if pretrained:
            self.cnn_encoder = EfficientNet.from_pretrained(model_name)
        else:
            self.cnn_encoder = EfficientNet.from_name(model_name)

        self.lstm = nn.LSTM(
            cnn_last_size, self.hidden_size, num_layers=3, bias=True,
            batch_first=True, bidirectional=True, dropout=0.5
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # for bidirectional, hidden_size * 2
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, images):
        batch_size, num_channels, time, height, width = images.shape
        # (h, c) = (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))
        hidden = None  # default to zero, the same as above
        embed = []
        for t in range(time):
            # with torch.no_grad():
            inp = images[:, :, t, :, :].squeeze(2)  # (-1, 3, t, 300, 300) -> (-1, 3, 300, 300)
            x = self.cnn_encoder.extract_features(inp)  # (-1, 3, 300, 300) -> (-1, 1536, 7, 7)
            x = self.avg_pooling(x)  # (-1, 1536, 7, 7) -> (-1, 1536, 1, 1)
            x = x.flatten(start_dim=1)  # (-1, 1536, 1, 1) -> (-1, 1536)
            embed.append(x)  # (time, -1, 1536)
        embed = torch.stack(embed, dim=0)
        # we need to permute the time dimension since batch_first=True will set batch as the 1st dimension
        embed = embed.permute(1, 0, 2)
        out, _ = self.lstm(embed, hidden)  # (-1, time, 1536) -> (-1, time, hidden_size*direction)

        out = torch.mean(out, dim=1, keepdim=True).squeeze(1)  # (-1, time, 300*2) -> (-1, 600)
        x = self.fc(out)  # (-1, 600) -> (-1, num_classes)
        return x


class TruncEfficientLSTMV3(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True, freeze=True):
        super(TruncEfficientLSTMV3, self).__init__()

        # efficientnetb3
        cnn_last_size = 1536
        self.hidden_size = 300
        if pretrained:
            self.cnn_encoder = EfficientNet.from_pretrained(model_name)
        else:
            self.cnn_encoder = EfficientNet.from_name(model_name)
        self._epochs = 3 if freeze else 0

        self.lstm = nn.LSTM(
            cnn_last_size, self.hidden_size, num_layers=3, bias=True,
            batch_first=True, bidirectional=True, dropout=0.5
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # for bidirectional, hidden_size * 2
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, images):
        batch_size, num_channels, time, height, width = images.shape
        # (h, c) = (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))
        hidden = None  # default to zero, the same as above
        embed = []
        for t in range(time):
            if self._epochs > 0:
                self._epochs -= 1
                with torch.no_grad():
                    inp = images[:, :, t, :, :].squeeze(2)  # (-1, 3, t, 300, 300) -> (-1, 3, 300, 300)
                    x = self.cnn_encoder.extract_features(inp)  # (-1, 3, 300, 300) -> (-1, 1536, 7, 7)
                    x = self.avg_pooling(x)  # (-1, 1536, 7, 7) -> (-1, 1536, 1, 1)
                    x = x.flatten(start_dim=1)  # (-1, 1536, 1, 1) -> (-1, 1536)
                    embed.append(x)  # (time, -1, 1536)
            else:
                inp = images[:, :, t, :, :].squeeze(2)  # (-1, 3, t, 300, 300) -> (-1, 3, 300, 300)
                x = self.cnn_encoder.extract_features(inp)  # (-1, 3, 300, 300) -> (-1, 1536, 7, 7)
                x = self.avg_pooling(x)  # (-1, 1536, 7, 7) -> (-1, 1536, 1, 1)
                x = x.flatten(start_dim=1)  # (-1, 1536, 1, 1) -> (-1, 1536)
                embed.append(x)  # (time, -1, 1536)
        embed = torch.stack(embed, dim=0)
        # we need to permute the time dimension since batch_first=True will set batch as the 1st dimension
        embed = embed.permute(1, 0, 2)
        out, _ = self.lstm(embed, hidden)  # (-1, time, 1536) -> (-1, time, hidden_size*direction)

        out = torch.mean(out, dim=1, keepdim=True).squeeze(1)  # (-1, time, 300*2) -> (-1, 600)
        x = self.fc(out)  # (-1, 600) -> (-1, num_classes)
        return x


if __name__ == "__main__":
    # hyper parameters
    cnn_model_name = "efficientnet-b3"
    img_dim, img_height, img_width = 3, 300, 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TruncEfficientLSTM(cnn_model_name, num_classes=1, pretrained=True).to(device)

    print(model)

    model_starts = summary(model, input_size=(img_dim, 7, img_height, img_width))
    summary_str = str(model_starts)
    with io.open("../log/trunc_efficient.txt", "w", encoding="utf-8") as f:
        f.write(summary_str)
