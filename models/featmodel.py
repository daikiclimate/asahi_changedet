import torch
import torch.nn as nn
import torchvision.models as models


class classifier(nn.Module):
    def __init__(self, input_channel=512, align_size=8):
        super(classifier, self).__init__()
        self.AdaptivePool = nn.AdaptiveAvgPool2d((align_size, align_size))
        self.hidden1 = 100
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, self.hidden1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden1),
            nn.ReLU(inplace=True),
        )
        self.hidden2 = 10
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.hidden1, self.hidden2, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.hidden2),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.hidden2 * 8 * 8, self.hidden1)
        self.activ = nn.ReLU()
        nn.init.kaiming_normal_(self.fc.weight)

        self.fc2 = nn.Linear(self.hidden1, 1)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.AdaptivePool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.hidden2 * 8 * 8)
        x = self.fc(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class ImgModel(nn.Module):
    def __init__(self, input_channel=3):
        super(ImgModel, self).__init__()
        model = models.resnet18(pretrained=True)
        self.back = torch.nn.Sequential(*(list(model.children())[:-1]))
        if input_channel != 3:
            self.back[0] = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)

        # self.back = self.back.features
        # self.vgg16 = models.vgg16(pretrained=True)
        # self.vgg16 = self.vgg16.features
        # self.classifier = classifier(input_channel=input_channel, align_size=align_size)
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, x, device):
        x = x["diff_img"]
        img = []
        for k in x.keys():
            img.append(x[k])
        x = torch.cat(img, 1)
        bs = x.shape[0]
        x = x.to(device)
        x = self.back(x)
        # x = self.classifier(x)
        x = x.view(bs, -1)
        x = self.fc(x)
        return x


class MultiHeadModel(nn.Module):
    def __init__(self, input_channel=3):
        super(MultiHeadModel, self).__init__()
        model = models.resnet18(pretrained=True)
        self.back = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.back = self.back[1:]
        # self.back[0] = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)
        # self.fc = nn.Sequential(
        #     nn.Linear(512, 100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(),
        #     nn.Linear(100, 2),
        # )
        f = 512
        self._aero_diff_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self._aero_diff_fc = FC(f)
        self._aero_src_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self._aero_src_fc = FC(f * 2)
        self._ele_diff_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self._ele_diff_fc = FC(f)
        self._ele_src_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self._ele_src_fc = FC(f * 2)
        self._final_fc = FC(f * 6, 100)
        # self._final_fc = FC(f*4, 100)

    def forward(self, x, device):
        diff_img = x["diff_img"]
        aero_diff_img = diff_img["aero"].to(device)
        bs = aero_diff_img.shape[0]
        aero_diff_img = self._aero_diff_conv(aero_diff_img)
        ele_diff_img = diff_img["ele"].to(device)
        ele_diff_img = self._ele_diff_conv(ele_diff_img)

        src_img = x["src_img"]
        aero_new_img = src_img["aero_new"].to(device)
        aero_old_img = src_img["aero_old"].to(device)
        aero_img = torch.cat([aero_new_img, aero_old_img])
        aero_img = self._aero_src_conv(aero_img)
        ele_new_img = src_img["ele_new"].to(device)
        ele_old_img = src_img["ele_old"].to(device)
        ele_img = torch.cat([ele_new_img, ele_old_img])
        ele_img = self._ele_src_conv(ele_img)
        x = torch.cat([aero_diff_img, ele_diff_img, aero_img, ele_img])
        x = self.back(x)
        aero_diff_feat = x[:bs].reshape(bs, -1)
        ele_diff_feat = x[bs : 2 * bs].reshape(bs, -1)
        aero_src_feat = x[2 * bs : 4 * bs]
        ele_src_feat = x[4 * bs :]
        aero_diff_out = self._aero_diff_fc(aero_diff_feat.reshape(bs, -1)).reshape(-1)
        ele_diff_out = self._ele_diff_fc(ele_diff_feat.reshape(bs, -1)).reshape(-1)
        aero_src_feat = torch.cat([aero_src_feat[:bs], aero_src_feat[bs:]], 1).reshape(
            bs, -1
        )
        aero_src_out = self._aero_src_fc(aero_src_feat).reshape(-1)
        ele_src_feat = torch.cat([ele_src_feat[:bs], ele_src_feat[bs:]], 1).reshape(
            bs, -1
        )
        ele_src_out = self._ele_src_fc(ele_src_feat).reshape(-1)
        feature = torch.cat(
            [aero_diff_feat, ele_diff_feat, aero_src_feat, ele_src_feat], 1
        )
        x = self._final_fc(feature).reshape(-1)
        return {
            "out": x,
            "aero_diff_out": aero_diff_out,
            "ele_diff_out": ele_diff_out,
            "aero_src_out": aero_src_out,
            "ele_src_out": ele_src_out,
        }


class FC(nn.Module):
    def __init__(self, input_channel=512, hidden=20, output=1):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_channel, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, output),
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    t = torch.tensor([0 for _ in range(512 * 8 * 8)], dtype=torch.float).reshape(
        1, 512, 8, 8
    )
    t = torch.tensor([0 for _ in range(512 * 9 * 9)], dtype=torch.float).reshape(
        1, 512, 9, 9
    )
    m = FeatModel()
    # m = ImgModel()
    print(m)
    # print(m(t).shape)
