import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune
from xtransfer.tools import mmc
from xtransfer.hook import add_to_dict
import math


class Conv(nn.Module):
    """
    output size calculation follows
    http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html

    """

    def __init__(self, dim_in=512, dim_out=512, input_size=None, **kwargs):
        super(Conv, self).__init__()
        torch.manual_seed(15)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.input_size = input_size
        self.ks = 3
        self.stride = 1
        self.pad = 1
        self.build_encoder()

    def build_encoder(self):
        self.conv = nn.Conv2d(in_channels=self.dim_in, out_channels=self.dim_in * 2, kernel_size=self.ks,
                              stride=self.stride, padding=self.pad)
        self.bn = nn.BatchNorm2d(num_features=self.dim_in * 2)
        self.relu = nn.ReLU(inplace=True)
        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.dim_in * 2 * self.input_size * self.input_size, self.dim_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Linear(nn.Module):
    def __init__(self, dim_in=512, dim_out=512, is_1d=False):
        super(Linear, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        if is_1d:
            self.adaptive_avg_func = F.adaptive_avg_pool1d
        else:
            self.adaptive_avg_func = F.adaptive_avg_pool2d

    def forward(self, x):
        x = self.adaptive_avg_func(x, 1)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, dim=-1)
        out = self.linear(x)
        return out


class AutoEncoderOG(nn.Module):
    """
        output size calculation follows
        http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html

    """

    def __init__(self, dim_in=512, dim_out=512, input_size=10, output_size=10, first_layer=False, **kwargs):
        super(AutoEncoderOG, self).__init__()
        torch.manual_seed(15)
        backbone_input = kwargs['backbone_input']
        self.pre_resizer = None

        head = kwargs['head']
        conv = self.find_conv(head)
        if isinstance(conv, nn.Conv1d):
            self.is_1d = True
        else:
            self.is_1d = False

        if conv:
            self.ks = conv.kernel_size
            self.stride = conv.stride
        else:
            self.is_1d = True
            self.ks = (3,)
            self.stride = (1,)

        if first_layer and not self.is_1d:
            self.pre_resizer = PreResizer(dim_in, backbone_input, input_size=input_size)
            self.dim_in = 1
            self.dim_out = 3
            self.size_i = backbone_input
            self.size_o = backbone_input
        else:
            self.dim_in = dim_in
            self.dim_out = dim_out
            self.size_i = input_size
            self.size_o = output_size

        self.dilation_size = 3 if self.size_i / (self.ks[0] * 3) > 3 else 1

        if self.size_i in [224, 84, 32, 100]:
            self.dim_inter = (self.dim_in + self.dim_out)
        else:
            self.dim_inter = math.ceil(self.dim_out / 2)

        if self.is_1d:
            self.build_encoder1d()
            self.build_decoder1d()
        else:
            self.build_encoder()
            self.build_decoder()

    def find_conv(self, model):
        objs = (torch.nn.Conv2d, torch.nn.Conv1d)
        for key, item in model.named_modules():
            if isinstance(item, objs):
                return item
        return None

    def build_encoder(self):
        unit = math.ceil(self.dim_inter - self.dim_in)
        self.encoder = nn.Conv2d(in_channels=self.dim_in, out_channels=self.dim_inter, kernel_size=self.ks,
                                 dilation=self.dilation_size, stride=self.stride)
        self.bn_i = nn.BatchNorm2d(num_features=self.dim_in + unit)

    def build_decoder(self):
        unit = math.ceil(self.dim_inter - self.dim_out)
        self.decoder = nn.ConvTranspose2d(in_channels=self.dim_inter, out_channels=self.dim_out,
                                          dilation=self.dilation_size,
                                          kernel_size=self.ks, stride=self.stride)
        self.bn_o = nn.BatchNorm2d(num_features=self.dim_inter - unit)

    def build_encoder1d(self):
        unit = math.ceil(self.dim_inter - self.dim_in)
        self.encoder = nn.Conv1d(in_channels=self.dim_in, out_channels=self.dim_inter, kernel_size=self.ks,
                                 dilation=self.dilation_size, stride=self.stride)
        self.bn_i = nn.BatchNorm1d(num_features=self.dim_in + unit)

    def build_decoder1d(self):
        unit = math.ceil(self.dim_inter - self.dim_out)
        self.decoder = nn.ConvTranspose1d(in_channels=self.dim_inter, out_channels=self.dim_out,
                                          dilation=self.dilation_size, kernel_size=self.ks, stride=self.stride)
        self.bn_o = nn.BatchNorm1d(num_features=self.dim_inter - unit)

    def forward(self, x):
        if self.pre_resizer is not None:
            x = self.pre_resizer(x)
        x = self.encoder(x)
        x = self.bn_i(x)

        if self.decoder is not None:
            x = self.decoder(x)
            x = self.bn_o(x)
        return x


class Resizer(nn.Module):
    def __init__(self, scale_factor, is_1d=False):
        super(Resizer, self).__init__()
        self.scale_factor = scale_factor
        self.is_1d = is_1d

    def forward(self, x):
        if self.is_1d:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=False)
        else:
            x = F.interpolate(x, size=(self.scale_factor, self.scale_factor), mode='bilinear', align_corners=False)
        return x


class PreResizer(nn.Module):
    def __init__(self, dim_in=512, backbone_input=32, **kwargs):
        super(PreResizer, self).__init__()
        torch.manual_seed(15)
        self.dim_in = dim_in
        self.backbone_input = backbone_input
        self.b_input = backbone_input
        self.input_size = kwargs['input_size']
        self.resizer = Resizer(backbone_input)

        if backbone_input in [224, 84]:
            self.kernel = 7
        else:
            self.kernel = 3

        if self.input_size < self.backbone_input:
            self.conv = None
            self.bn = None
            self.output_size = self.input_size
            self.out_channels = self.dim_in
        else:
            # self.dilation = 1
            # self.stride = 1
            # self.kernel = 5
            self.calculate_stride()
            self.calculate_dilation()
            self.build_layers()
            self.output_size = int(self.calculate_out_size())

        # p_topbottom = self.backbone_input - self.out_channels
        # p_top = p_topbottom//2
        # p_bottom = p_topbottom - p_top
        #
        # p_leftright = self.backbone_input - self.output_size
        # p_left = p_leftright//2
        # p_right = p_leftright - p_left
        # self.pad = nn.ZeroPad2d((p_left, p_right, p_top, p_bottom))

    def calculate_stride(self):
        self.stride = int(self.input_size / self.backbone_input)

    def calculate_paddings(self):
        return 0

    def calculate_out_size(self):
        sout = (self.input_size - self.dilation * (self.kernel - 1) - 1) / self.stride + 1
        return sout

    def calculate_dilation(self):
        d = (self.input_size - (self.backbone_input - 1) * self.stride - 1) / (self.kernel - 1)
        self.dilation = int(d) + 1

    def build_layers(self):
        # out_channels_addition = self.input_size - self.backbone_input
        # out_channels_addition = 0
        # self.out_channels = min(self.backbone_input, self.dim_in**2)
        self.out_channels = self.backbone_input
        self.conv = nn.Conv1d(in_channels=self.dim_in, out_channels=self.out_channels, kernel_size=self.kernel,
                              stride=self.stride, dilation=self.dilation)
        self.bn = nn.BatchNorm1d(num_features=self.out_channels)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
            x = self.bn(x)
        x = x.unsqueeze(1)
        x = self.resizer(x)
        # x = self.pad(x)
        return x


class TopK(nn.Module):
    def __init__(self, mask):
        super(TopK, self).__init__()
        self.register_buffer('mask', mask)

    def forward(self, x):
        out = x * self.mask
        return out


class Trainer_Npair(nn.Module):
    def __init__(self, anchor_pca, anchor_mean, dim_in, dim_out, input_size,
                 output_size, head, **kwargs):
        super(Trainer_Npair, self).__init__()
        self.anchor_pca = nn.Parameter(anchor_pca, requires_grad=False)
        self.anchor_mean = nn.Parameter(anchor_mean, requires_grad=False)
        if 'model' in kwargs and kwargs['model'] is not None:
            self.model = kwargs['model']
        else:
            self.model = AutoEncoderOG(dim_in, dim_out, input_size, output_size)
        # self.model = AutoEncoderD(dim_in, dim_out, input_size, output_size)
        self.norm_mode = kwargs['norm_mode']
        self.head = head
        self.scale = None

    def set_scale(self, scale):
        self.scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.mmc(x)
        if 'S' in self.norm_mode:
            x = F.normalize(x)
        return x

    def mmc(self, feature):
        if not torch.is_tensor(feature):
            feature = torch.from_numpy(feature)
        if feature.dim() > 4:
            shape = list([-1] + list(feature.shape[2:]))
            feature = feature.view(shape)
        feature = F.adaptive_avg_pool2d(feature, 1).squeeze_(-1).squeeze_(-1)
        return feature

    def init_weights(self, mean=0., std=1.):
        def init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(m.weight, mean, std)
                # torch.nn.init.uniform_(m.weight, a=-1, b=std)

        self.model.apply(init)


class Trainer_RotationMatrix(nn.Module):
    def __init__(self, num_channels, input, anchor, **kwargs):
        super(Trainer_RotationMatrix, self).__init__()
        torch.manual_seed(5)
        self.layer_id = kwargs['layer_id']
        self.input = torch.from_numpy(input).float()
        self.input = self.input / torch.linalg.norm(self.input)
        self.anchor = torch.from_numpy(anchor).float()
        self.anchor = self.anchor / torch.linalg.norm(self.anchor)
        # rotation_matrix = torch.from_numpy(np.identity(num_channels)).float()

        rotation_matrix = torch.ones((num_channels, num_channels))
        torch.nn.init.uniform_(rotation_matrix, -1.0, 1.0).float()

        self.rotation_matrix = nn.Parameter(rotation_matrix, requires_grad=True)
        self.optimizer = torch.optim.SGD(params=[self.rotation_matrix], lr=0.01, momentum=0.95)
        self.loss_fun = torch.nn.CosineEmbeddingLoss()

    def train(self, epoch):
        # self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)
        for e in range(epoch):
            out = torch.mm(self.input, self.rotation_matrix)
            loss = self.loss_fun(out, self.anchor, torch.Tensor(out.size(0)).fill_(1.0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            if e == 0 or e == epoch - 1:
                print('Episode {:05} >>> Loss is: {:.5f}'.format(e + 1, loss.item()))
            # if (e + 1) % 5 == 0:
            #     rm = self.rotation_matrix
            #     rm = rm.detach()
            #     add_to_dict('RM{}_{}'.format(e + 1, self.layer_id), rm.numpy())

    def get_rm(self):
        self.rotation_matrix.requires_grad = False
        return self.rotation_matrix


class Trainer(nn.Module):
    def __init__(self, anchor_pca, anchor_mean, dim_in, dim_out, input_size,
                 output_size, head, **kwargs):
        super(Trainer, self).__init__()
        if anchor_mean is not None and anchor_pca is not None:
            self.anchor_pca = nn.Parameter(anchor_pca, requires_grad=False)
            self.anchor_mean = nn.Parameter(anchor_mean, requires_grad=False)
        else:
            self.anchor_mean = anchor_mean
            self.anchor_pca = anchor_pca

        if 'model' in kwargs and kwargs['model'] is not None:
            self.model = kwargs['model']
        else:
            first_layer = kwargs['first_layer']
            backbone_input = kwargs['backbone_input']
            self.model = AutoEncoderOG(dim_in, dim_out, input_size, output_size, head=head, first_layer=first_layer,
                                       backbone_input=backbone_input)
            # self.model = AutoEncoder(dim_in, dim_out, input_size, output_size)
        # self.model = AutoEncoderD(dim_in, dim_out, input_size, output_size)
        self.norm_mode = kwargs['norm_mode']
        self.head = head
        self.scale = None
        self.rm = None

    def set_scale(self, scale):
        self.scale = nn.Parameter(scale, requires_grad=False)

    def set_rm(self, rm):
        self.rm = nn.Parameter(rm.double(), requires_grad=False)

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = mmc(x)
        if 'S' in self.norm_mode:
            x = F.normalize(x)
        if self.anchor_mean is not None:
            x -= self.anchor_mean
            x = x @ self.anchor_pca
        if self.scale is not None:
            x = x * self.scale
        if self.rm is not None:
            x = torch.mm(x, self.rm)
        return x

    def init_weights(self, mean=0., std=1.):
        def init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(m.weight, mean, std)
                # torch.nn.init.uniform_(m.weight, a=-1, b=std)

        self.model.apply(init)


class TrainerNorm(nn.Module):
    def __init__(self, dim_in, dim_out, input_size, output_size, head, num_classes, **kwargs):
        super(TrainerNorm, self).__init__()
        self.num_classes = num_classes
        if 'model' in kwargs and kwargs['model'] is not None:
            self.model = kwargs['model']
        else:
            self.model = AutoEncoderOG(dim_in, dim_out, input_size, output_size)
        self.classifier = nn.Linear(kwargs['out_channels'], num_classes)
        self.norm_mode = kwargs['norm_mode']
        self.head = head

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.mmc(x)
        if 'S' in self.norm_mode:
            x = F.normalize(x)
        out = self.classifier(x)
        return out

    def mmc(self, feature):
        if not torch.is_tensor(feature):
            feature = torch.from_numpy(feature)
        if feature.dim() > 4:
            shape = list([-1] + list(feature.shape[2:]))
            feature = feature.view(shape)
        feature = F.adaptive_avg_pool2d(feature, 1).squeeze_(-1).squeeze_(-1)
        return feature


class TrainerCNN(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(TrainerCNN, self).__init__()
        self.model = Conv(dim_in, dim_out)
        self.norm_mode = kwargs['norm_mode']

    def forward(self, x):
        x = self.model(x)
        x = self.mmc(x)
        if 'S' in self.norm_mode:
            x = F.normalize(x)
        return x

    def mmc(self, feature):
        if not torch.is_tensor(feature):
            feature = torch.from_numpy(feature)
        if feature.dim() > 4:
            shape = list([-1] + list(feature.shape[2:]))
            feature = feature.view(shape)
        feature = F.adaptive_avg_pool2d(feature, 1).squeeze_(-1).squeeze_(-1)
        return feature


class TrainerMMC(nn.Module):
    def __init__(self, dim_in, dim_out, input_size, output_size, head):
        super(TrainerMMC, self).__init__()
        self.model = AutoEncoderOG(dim_in, dim_out, input_size, output_size)
        self.head = head
        self.scale = None

    def forward(self, x):
        x = self.model(x)
        # x = F.normalize(x)
        x = self.head(x)
        out = self.mmc(x)

        if self.scale is not None:
            out = out * self.scale
        return out

    def mmc(self, feature):
        if not torch.is_tensor(feature):
            feature = torch.from_numpy(feature)
        if feature.dim() > 4:
            shape = list([-1] + list(feature.shape[2:]))
            feature = feature.view(shape)
        feature = F.adaptive_avg_pool2d(feature, 1).squeeze_(-1).squeeze_(-1)
        return feature

    def set_scale(self, scale):
        scale = torch.from_numpy(scale)
        self.scale = nn.Parameter(scale, requires_grad=False)


class AutoEncoderPCA(nn.Module):
    """
    output size calculation follows
    http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html

    """

    def __init__(self, pca, mean, dim_out=512, input_size=10, output_size=10):
        super(AutoEncoderPCA, self).__init__()
        self.register_buffer('pca', pca)
        self.register_buffer('mean', mean[None, None, None, :])
        self.dim_in = pca.size(1)
        self.dim_out = dim_out
        self.size_i = input_size
        self.size_o = output_size
        self.dim_inter = (self.dim_in + self.dim_out) // 2
        self.set_kernel_size()
        self.build_encoder()
        self.bn_i = nn.BatchNorm2d(num_features=self.dim_inter)
        self.build_decoder()
        self.bn_o = nn.BatchNorm2d(num_features=self.dim_out)

    def build_encoder(self):
        self.encoder = nn.Conv2d(in_channels=self.dim_in, out_channels=self.dim_inter, kernel_size=self.ks_i)

    def build_decoder(self):
        self.decoder = nn.ConvTranspose2d(in_channels=self.dim_inter, out_channels=self.dim_out,
                                          kernel_size=self.ks_o, stride=(1, 1), padding=(0, 0))

    def set_kernel_size(self):
        if self.size_i == self.size_o:
            self.ks_i = 5
            self.ks_o = 5

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        x = x - self.mean
        x = x @ self.pca
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.encoder(x)
        x = self.bn_i(x)
        x = self.decoder(x)
        x = self.bn_o(x)
        return x


def get_weight_dist(model):
    ms = []
    stds = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mean = torch.mean(m.weight)
            std = torch.std(m.weight)
            ms.append(mean.item())
            stds.append(std.item())
    return np.mean(ms), np.mean(stds)


class AutoEncoder(nn.Module):
    """
    output size calculation follows
    http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html

    """

    def __init__(self, dim_in=512, dim_out=512, input_size=10, output_size=10, first_layer=False, **kwargs):
        super(AutoEncoder, self).__init__()
        torch.manual_seed(15)
        backbone_input = kwargs['backbone_input']
        self.pre_resizer = None
        if first_layer:
            self.pre_resizer = PreResizer(dim_in, backbone_input, input_size=input_size)
            self.dim_in = 1
            self.dim_out = 3
            self.size_i = backbone_input
            self.size_o = backbone_input
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.size_i = input_size
        self.size_o = output_size
        self.dim_inter = (self.dim_in + self.dim_out)
        self.padding = None

        if input_size == 6:
            self.dim_inter = (self.dim_in + self.dim_out)
            self.build_encoder_raw()
            self.build_decoder_raw()
        elif input_size == 12:
            self.dim_inter = (self.dim_in + self.dim_out)
            self.build_encoder_stft()
            self.build_decoder_stft()
        else:
            self.dim_inter = math.ceil(self.dim_out / 1.5)
            self.set_kernel()
            self.build_encoder()
            self.build_decoder()

    def build_encoder_raw(self):
        self.encoder = nn.Conv2d(in_channels=self.dim_in, out_channels=self.dim_inter, kernel_size=(1, 7),
                                 dilation=(1, 7), stride=(1, 1))
        self.bn_i = nn.BatchNorm2d(num_features=self.dim_inter)

    def build_decoder_raw(self):
        self.decoder = nn.ConvTranspose2d(in_channels=self.dim_inter, out_channels=self.dim_out,
                                          dilation=(10, 2), kernel_size=(23, 6), stride=(1, 1))
        self.padding = nn.ZeroPad2d((0, 0, 1, 2))
        self.bn_o = nn.BatchNorm2d(num_features=self.dim_out)

    def build_encoder_stft(self):
        self.encoder = nn.Conv2d(in_channels=self.dim_in, out_channels=self.dim_inter, kernel_size=(3, 3),
                                 dilation=(1, 1), stride=(1, 1))
        self.bn_i = nn.BatchNorm2d(num_features=self.dim_inter)

    def build_decoder_stft(self):
        self.decoder = nn.ConvTranspose2d(in_channels=self.dim_inter, out_channels=self.dim_out,
                                          dilation=(21, 21), kernel_size=(11, 11), stride=(1, 1))
        self.padding = nn.ZeroPad2d((1, 2, 2, 2))
        self.bn_o = nn.BatchNorm2d(num_features=self.dim_out)

    def set_kernel(self):
        self.pad = None
        if self.size_i == self.size_o:
            self.k_in = 3
            self.k_out = 3
            self.d_in = 1
            self.d_out = 1
            self.s_in = 1
            self.s_out = 1
        else:
            self.size_inter = self.size_o // 2
            self.s_in = int(self.size_i / self.size_inter)
            self.k_in = 3
            self.d_in = 2
            self.k_out = 3
            self.s_out = 1
            self.d_out = (self.size_o - self.size_inter) / (self.k_out - 1)
            if self.d_out != int(self.d_out):
                self.pad = int(self.size_inter - int(self.d_out) * (self.k_out - 1))
            self.d_out = int(self.d_out)

    def build_encoder(self):
        self.encoder = nn.Conv2d(in_channels=self.dim_in, out_channels=self.dim_inter, kernel_size=self.k_in,
                                 dilation=self.d_in, stride=self.s_in, padding=1)
        self.bn_i = nn.BatchNorm2d(num_features=self.dim_inter)

    def build_decoder(self):
        self.decoder = nn.ConvTranspose2d(in_channels=self.dim_inter, out_channels=self.dim_out,
                                          dilation=self.d_out, kernel_size=self.k_out, stride=self.s_out)
        self.bn_o = nn.BatchNorm2d(num_features=self.dim_out)
        if self.pad is not None:
            self.padding = nn.ZeroPad2d((0, self.pad, 0, self.pad))

    def forward(self, x):
        if self.pre_resizer is not None:
            x = self.pre_resizer(x)
        x = self.encoder(x)
        x = self.bn_i(x)
        x = self.decoder(x)
        if self.padding is not None:
            x = self.padding(x)
        x = self.bn_o(x)
        return x


if __name__ == "__main__":
    import torch
    import numpy as np

    model = PreResizer(dim_in=6, backbone_input=32, input_size=256)
    input = torch.rand(2, 6, 256)

    output = model(input)
    print(output.size())

    # for s in [224, 56, 28, 14, 7]:
    # model.calculate_kenel_size(s)

    # pca = torch.randn(16, 2)
    # mean = torch.randn(16)
    # model = AutoEncoderChannel(pca, mean, dim_out, input_size, output_size)
    # model = AutoEncoderD(dim_in, dim_out, input_size, output_size)
    # input = torch.randn(2, 3, 224, 224)
    # output = model(input)
    # print(output.size())

    # input = torch.randn(10,5,6,84,84)
    # model = Header(num_channels=6)
    # output = model(input)
    # print(output.size())
