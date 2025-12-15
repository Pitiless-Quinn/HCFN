import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
import random
import numpy as np


class Conv2dL2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, weight_decay=0., bias=False):
        super(Conv2dL2, self).__init__()
        self.conv2 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               bias=bias,
                               groups=groups,
                               )
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.conv2(x)

    def l2_loss(self):
        return self.weight_decay * torch.sum(self.conv2.weight ** 2)


class MultiScaleConvBlock(nn.Module):
    def __init__(self, F1=4, D=2, in_chans=22, dropout=0.5,
                 kern_lengths=[32, 64, 128],
                 poolSize=8):
        super(MultiScaleConvBlock, self).__init__()

        self.temporal_convs = nn.ModuleList()
        for kern_length in kern_lengths:
            self.temporal_convs.append(
                Conv2dL2(1, F1, (kern_length, 1), padding='same', bias=False, weight_decay=0.009)
            )

        num_branches = len(kern_lengths)
        F1_multiplied = F1 * num_branches
        F2 = F1_multiplied * D

        self.batchnorm1 = nn.BatchNorm2d(F1_multiplied)
        self.depthwise = Conv2dL2(F1_multiplied, F1_multiplied * D, (1, in_chans), groups=F1_multiplied, bias=False,
                                  weight_decay=0.009)
        self.batchnorm2 = nn.BatchNorm2d(F2)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((8, 1))
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = Conv2dL2(F2, F2, (16, 1), padding='same', bias=False, weight_decay=0.009)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((poolSize, 1))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)

        branch_outputs = []
        for conv in self.temporal_convs:
            branch_outputs.append(conv(x))

        x = torch.cat(branch_outputs, dim=1)

        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        return x


class FuzzyCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, n_rules=5, bias=True, controller=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_rules = n_rules

        self.padding = (kernel_size - 1) * dilation

        self.base_kernels = nn.Parameter(
            torch.randn(n_rules, out_channels, in_channels, kernel_size)
        )
        self.base_biases = nn.Parameter(torch.randn(n_rules, out_channels)) if bias else None

        Init.kaiming_uniform_(self.base_kernels, a=np.sqrt(5))
        if self.base_biases is not None:
            fan_in, _ = Init._calculate_fan_in_and_fan_out(self.base_kernels[0])
            bound = 1 / np.sqrt(fan_in)
            Init.uniform_(self.base_biases, -bound, bound)

        self.controller = controller

    def forward(self, x):
        batch_size, _, length = x.shape
        padded_x = F.pad(x, (self.padding, 0))
        patches = padded_x.unfold(dimension=2, size=self.kernel_size, step=1)
        patches = patches[:, :, :length, :]
        patches = patches.transpose(1, 2)
        patches = patches.reshape(batch_size, length, self.in_channels * self.kernel_size)
        firing_strengths = self.controller(patches)  # -> [batch, length, n_rules]
        expert_outputs = []
        for i in range(self.n_rules):
            kernel_i = self.base_kernels[i]
            bias_i = self.base_biases[i] if self.base_biases is not None else None
            output_i = F.conv1d(padded_x, kernel_i, bias_i, dilation=self.dilation)
            expert_outputs.append(output_i)
        all_outputs = torch.stack(expert_outputs, dim=-1)
        frs_reshaped = firing_strengths.unsqueeze(1)
        final_output = torch.sum(all_outputs * frs_reshaped, dim=-1)
        return final_output


class AFTCNBlock(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout, tcn_n_rules=5, activation=nn.ELU,
                 controller_hidden_dim=16):
        super().__init__()
        self.activation = activation()
        self.dropout = dropout
        self.depth = depth
        self.filters = filters
        self.kernel_size = kernel_size

        self.layers = nn.ModuleList()
        self.downsample = (
            nn.Conv1d(input_dimension, filters, kernel_size=1, bias=False)
            if input_dimension != filters
            else None
        )

        patch_dim_input = input_dimension * kernel_size
        self.controller_for_input_dim = nn.Sequential(
            nn.Linear(patch_dim_input, controller_hidden_dim),
            nn.ReLU(),
            nn.Linear(controller_hidden_dim, tcn_n_rules),
            nn.Softmax(dim=-1)
        )

        patch_dim_filters = filters * kernel_size
        self.controller_for_filters_dim = nn.Sequential(
            nn.Linear(patch_dim_filters, controller_hidden_dim),
            nn.ReLU(),
            nn.Linear(controller_hidden_dim, tcn_n_rules),
            nn.Softmax(dim=-1)
        )

        for i in range(depth):
            dilation = 2 ** i

            first_conv_controller = self.controller_for_input_dim if i == 0 else self.controller_for_filters_dim

            conv_block = nn.Sequential(
                FuzzyCausalConv1d(
                    in_channels=input_dimension if i == 0 else filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    n_rules=tcn_n_rules,
                    bias=True,
                    controller=first_conv_controller
                ),
                nn.BatchNorm1d(filters),
                self.activation,
                nn.Dropout(self.dropout),
                FuzzyCausalConv1d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    n_rules=tcn_n_rules,
                    bias=True,
                    controller=self.controller_for_filters_dim
                ),
                nn.BatchNorm1d(filters),
                self.activation,
                nn.Dropout(self.dropout),
            )
            self.layers.append(conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        res = x if self.downsample is None else self.downsample(x)
        for index, layer in enumerate(self.layers):
            out = layer(x)
            out = out + res
            out = self.activation(out)
            res = out
            x = out

        out = out.permute(0, 2, 1)
        return out


class FuzzyAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FuzzyAttention, self).__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.context_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, fuzzy_context):
        Q = self.query(x)  # [batch, time, features]
        K = self.key(x)  # [batch, time, features]
        V = self.value(x)  # [batch, time, features]

        guided_Q = Q + self.context_projection(fuzzy_context.unsqueeze(1))

        attn_scores = torch.matmul(guided_Q, K.transpose(1, 2)) / (self.feature_dim ** 0.5)
        attn_weights = self.softmax(attn_scores)

        attended = torch.matmul(attn_weights, V)
        return torch.mean(attended, dim=1)  # [batch, features]


class TSK(nn.Module):
    def __init__(self, in_dim, n_rules, n_classes, init_centers, init_Vs, ampli):
        super(TSK, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.init_centers = init_centers
        self.init_Vs = init_Vs
        self.ampli = ampli
        self.bn = nn.BatchNorm1d(num_features=self.in_dim)

        self.eps = 1e-8

        self.Cons = torch.FloatTensor(size=(self.n_rules, self.in_dim, self.n_classes))
        self.Bias = torch.FloatTensor(size=(1, self.n_rules, self.n_classes))
        self.Cs = torch.FloatTensor(size=(self.in_dim, self.n_rules))
        self.Vs = torch.FloatTensor(size=self.Cs.size())

        self.Cons = nn.Parameter(self.Cons, requires_grad=True)
        self.Bias = nn.Parameter(self.Bias, requires_grad=True)
        self.Cs = nn.Parameter(self.Cs, requires_grad=True)
        self.Vs = nn.Parameter(self.Vs, requires_grad=True)

        self.Cs.data = torch.from_numpy(self.init_centers).float()
        if self.init_Vs is not None:
            self.Vs.data = torch.from_numpy(self.init_Vs).float()
        else:
            Init.normal_(self.Vs, mean=1, std=0.2)
        Init.uniform_(self.Cons, -1, 1)
        Init.constant_(self.Bias, 0)

    def get_firing_strengths(self, x):
        frs = torch.exp(
            torch.sum(
                -(x.unsqueeze(dim=2) - self.Cs) ** 2 / (2 * self.Vs ** 2), dim=1
            ) + self.ampli
        )
        return frs / (torch.sum(frs, dim=1, keepdim=True) + self.eps)

    def classify_with_features(self, x, frs):
        x = self.bn(x)
        x_rep = x.unsqueeze(dim=1).expand([x.size(0), self.n_rules, x.size(1)])
        cons = torch.einsum('ijk,jkl->ijl', [x_rep, self.Cons])
        cons = cons + self.Bias
        cons = torch.mul(cons, frs.unsqueeze(2))
        return torch.sum(cons, dim=1, keepdim=False)

    def forward(self, x, with_frs=False):
        frs = self.get_firing_strengths(x)
        output = self.classify_with_features(x, frs)
        if with_frs:
            return output, frs
        return output

    def l2_loss(self):
        return torch.sum(self.Cons ** 2)

    def ur_loss(self, frs):
        return ((torch.mean(frs, dim=0) - 1/self.n_rules)**2).sum()


class HCFN(nn.Module):
    def __init__(self, in_chans=22, n_classes=4,
                 eegn_F1=8, eegn_D=2, eegn_kernLengths=[32, 64, 96], eegn_poolSize=7, eegn_dropout=0.4,
                 tcn_depth=2, tcn_kernelSize=4, tcn_dropout=0.35, tcn_n_rules=2,
                 tsk_n_rules=14, tsk_initCenters=None, tsk_initVs=None, tsk_ampli=0):

        super(HCFN , self).__init__()

        self.conv_block = MultiScaleConvBlock(
            F1=eegn_F1,
            D=eegn_D,
            kern_lengths=eegn_kernLengths,
            poolSize=eegn_poolSize,
            in_chans=in_chans,
            dropout=eegn_dropout
        )

        num_branches = len(eegn_kernLengths)
        F2 = eegn_F1 * num_branches * eegn_D
        tcn_input_dim = F2
        tcn_filters = F2

        self.aftcn = AFTCNBlock(
            input_dimension=tcn_input_dim,
            depth=tcn_depth,
            kernel_size=tcn_kernelSize,
            filters=tcn_filters,
            dropout=tcn_dropout,
            tcn_n_rules=tcn_n_rules
        )

        tsk_indim = tcn_filters
        self.attention = FuzzyAttention(feature_dim=tsk_indim)

        tsk_initCenters = np.random.randn(tsk_indim, tsk_n_rules)
        self.classifier = TSK(
            in_dim=tsk_indim,
            n_rules=tsk_n_rules,
            n_classes=n_classes,
            init_centers=tsk_initCenters,
            init_Vs=tsk_initVs,
            ampli=tsk_ampli
        )

    def forward(self, x, with_frs=False):
        x = self.conv_block(x)  # -> [batch, F2, time, 1]
        x = x[:, :, :, 0].permute(0, 2, 1)  # -> [batch, time, F2]
        time_series_features = self.aftcn(x)  # -> [batch, time, tcn_filters]
        preliminary_features = torch.mean(time_series_features, dim=1)
        frs = self.classifier.get_firing_strengths(preliminary_features)  # -> [batch, n_rules]
        fuzzy_context = torch.matmul(frs, self.classifier.Cs.T)  # -> [batch, features]
        final_features = self.attention(time_series_features, fuzzy_context)
        output = self.classifier.classify_with_features(final_features, frs)
        if with_frs:
            return output, frs
        return output


    def total_loss(self, output, target, frs, ce_weight=1.0, l2_weight=0.001, ur_weight=0.1):
        ce_loss = F.cross_entropy(output, target)
        l2_reg = 0.0
        for model in self.modules():
            if hasattr(model, 'l2_loss'):
                l2_reg = l2_reg + model.l2_loss()
        ur_reg = self.classifier.ur_loss(frs)

        total_loss = ce_weight * ce_loss + \
                     l2_weight * l2_reg + \
                     ur_weight * ur_reg

        return total_loss


