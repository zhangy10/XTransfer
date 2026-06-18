import math
import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from thop import profile
from thop import clever_format
import numpy as np
import time
from scipy.optimize import curve_fit
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from xtransfer.encoder import Linear, Resizer
import torch.nn.utils.prune as prune

np.set_printoptions(precision=4)

import xtransfer.torch_pruning as tp
from torch.optim.lr_scheduler import StepLR
from xtransfer.tools import mmc, intra_distance, pca_fit, class_centroids, map_classes_best, \
    class_centroids_sort
from xtransfer.encoder import Trainer, TopK, TrainerNorm, Trainer_Npair, TrainerCNN, AutoEncoderOG
from xtransfer.sampling_losses import CrossSample, AnchorLoss, PositiveNegativeLoss, MMD
from xtransfer.metric_losses import TripletLoss, NPairLoss
from xtransfer.tools import class_silhouette_score, select_topN, inter_distance, get_next_key, build_model_dict, \
    bn_modification, class_centroids_dict
from xtransfer.hook import add_to_dict, get_mask_hook, get_activation_hook, _remove_all_forward_hooks
from modeling.backbone.resnet import BasicBlock
from modeling.backbone.resnet1d import BasicBlock as BasicBlock1d
from xtransfer.encoder import Trainer_RotationMatrix, Conv
# from xtransfer.encoder import AutoEncoder as AutoEncoderOG
from xtransfer.encoder import AutoEncoderOG
from xtransfer.engine import EarlyStopping
from utils import Notes, set_random_seed
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def func(x, a, b):
    return np.exp(a * x) + b


def fit_curve(data):
    popt, pcov = curve_fit(func, data['x'], data['y'])
    return popt


def calculate_anchor_score(dictL, norm_mode=None, num_classes=6, n_comp=2):
    anchor_x = dictL['anchor']
    anchor_y = dictL['anchor_label']
    ax_mmc = mmc(anchor_x)

    if 'I' in norm_mode:
        ax_mmc = F.normalize(ax_mmc)

    fitted_pca = pca_fit(ax_mmc, n_comp=n_comp, labels=anchor_y)

    ax_pca = fitted_pca.transform(ax_mmc)
    aScore = class_silhouette_score(ax_pca, anchor_y)
    aTOPN = select_topN(aScore, top_n=num_classes)
    anchor_score = np.mean([aScore[a] for a in aTOPN])
    return anchor_score


@torch.no_grad()
def quick_check(x, y, anchor_x, anchor_y=None, next_input_size=None, head=None, model=None, n_comp=2, regression=False,
                **kwargs):
    norm_mode = kwargs['norm_mode']
    ax_mmc = mmc(anchor_x)
    if 'I' in norm_mode:
        ax_mmc = F.normalize(ax_mmc)
    fitted_pca = pca_fit(ax_mmc, n_comp=n_comp, labels=anchor_y)

    if model is None:
        first_layer = kwargs['first_layer']
        backbone_input = kwargs['backbone_input']
        dim_in = x.size(1)
        dim_out = next_input_size[0]
        input_size = x.size(2)
        output_size = next_input_size[1]
        model = AutoEncoderOG(dim_in, dim_out, input_size, output_size, head=head,
                              first_layer=first_layer, backbone_input=backbone_input)

    x = model(x)
    x = head(x)
    x_mmc = mmc(x).cpu()
    if 'S' in norm_mode:
        x_mmc = F.normalize(x_mmc)

    if 'prune_mask' in kwargs:
        prune_mask = kwargs['prune_mask']
        if prune_mask is not None:
            fitted_pca.components_ = fitted_pca.components_[:, prune_mask]
            fitted_pca.mean_ = fitted_pca.mean_[prune_mask]
            fitted_pca.n_features_in_ = len(prune_mask)
    x_pca = fitted_pca.transform(x_mmc)
    # if 'dic_key' in kwargs.keys():
    #     key = kwargs['dic_key']
    #     lid = kwargs['layer_id']
    #     add_to_dict('{}_{}'.format(key, lid), x_pca)
    class_score = class_silhouette_score(x_pca, y, 4, regression=regression)
    mean_score = np.round(np.mean(list(class_score.values())), 4)
    del x_mmc
    del x_pca
    print('PCA S-score for each class: {} , Mean:{:.4f}'.format(class_score, mean_score))
    return mean_score, model


@torch.no_grad()
def prototype_test(x, y, test_x, test_y, head, model, resizer, n_support=5, n_query=15):
    # n_way = len(np.unique(test_y))
    if resizer is not None:
        x = resizer(x)
        test_x = resizer(test_x)
    x_feature = F.normalize(mmc(head(model(x)))).cpu()
    proto = class_centroids_sort(x_feature, y)

    # train
    train_dists = euclidean_dist(x_feature, proto)
    train_scores = -train_dists
    topk_scores, topk_labels = train_scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    train_y = y.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == train_y)
    train_acc = top1_correct / len(train_y)

    # test
    x_query = F.normalize(mmc(head(model(test_x)))).cpu()
    # query = x_query.contiguous()
    dists = euclidean_dist(x_query, proto)
    scores = -dists

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    test_y = test_y.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == test_y)
    test_acc = top1_correct / len(test_y)
    return train_acc, test_acc


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


@torch.no_grad()
def bn_init(x, next_input_size, head, model, **kwargs):
    bn_modification(head, momentum=1, is_train=True)

    # dim_in = x.size(1)
    # dim_out = next_input_size[0]
    # input_size = x.size(2)
    # output_size = next_input_size[1]
    # model = AutoEncoderOG(dim_in, dim_out, input_size, output_size, head=head)

    x = model(x)
    head(x)

    bn_modification(head, momentum=0.1, is_train=True)
    # return head


class Downsample(nn.Module):
    """
    output size calculation follows
    http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html

    """

    def __init__(self, idx):
        super(Downsample, self).__init__()
        self.register_buffer('idx', idx.long())

    def forward(self, x):
        x = x[:, self.idx]
        return x


class ResizeTrans:
    def __init__(self, input_size, og_input_size, is_1d=False):
        if is_1d:
            scale_factor = og_input_size[-1] / input_size[-1]
        else:
            scale_factor = og_input_size[-1]
        self.resizer = Resizer(scale_factor, is_1d=is_1d)

    def get_resizer(self):
        return self.resizer


class MMCTrans:
    def __init__(self, x, y, anchor_x, anchor_y, head, num_episode=100, n_comp=2, **kwargs):
        self.training = True
        self.num_classes = len(torch.unique(y))
        self.x = x
        self.y = y
        self.head = head
        self.num_episode = num_episode
        self.num_episode_rot = kwargs['num_episode_rot']
        self.out_size = kwargs['out_size']
        self.n_comp = n_comp
        self.layer_id = kwargs['layer_id']
        self.mode = kwargs['mode']
        self.norm_mode = kwargs['norm_mode']
        self.regression = kwargs['regression']
        self.test_x = kwargs['test_x']
        self.test_y = kwargs['test_y']
        self.rotate = kwargs['rm']
        self.best_anchor_mode = kwargs['best_anchor']
        self.first_layer = kwargs['first_layer']
        self.backbone_input = kwargs['backbone_input']
        self.loss_mode = kwargs['loss_mode']
        if 'prehead' in kwargs:
            self.prehead = kwargs['prehead']
        else:
            self.prehead = None
        print("Numpy sampling seed: {}".format(np.random.get_state()[1][0]))

        if self.regression:
            self.le = preprocessing.LabelEncoder()
            self.le.fit(self.y)
            self.y = self.le.transform(self.y)
        self.update_param(x, y, anchor_x, anchor_y)

    def update_param(self, x, y, anchor_x, anchor_y):
        ax_mmc = mmc(anchor_x)

        if 'I' in self.norm_mode:
            ax_mmc = F.normalize(ax_mmc)

        fitted_pca = pca_fit(ax_mmc, n_comp=self.n_comp, labels=anchor_y)
        self.pca = fitted_pca
        ax_pca = fitted_pca.transform(ax_mmc)
        aScore = class_silhouette_score(ax_mmc, anchor_y, regression=self.regression)
        aTOPN = select_topN(aScore, top_n=self.num_classes)
        self.anchor_score = np.mean([aScore[a] for a in aTOPN])

        aMedoids = class_centroids(ax_mmc, anchor_y)

        anchors = []
        for i in range(self.num_classes):
            anchors.append(aMedoids[aTOPN[i]])
        anchors = np.stack(anchors)
        self.anchors = torch.from_numpy(anchors)

        anchor_pca = None
        anchor_mean = None

        # build trainer
        dim_in = self.x.size(1)
        dim_out = self.out_size[0]
        input_size = self.x.size(2)
        output_size = self.out_size[1]
        trainer = Trainer(anchor_pca, anchor_mean, dim_in, dim_out, input_size,
                          output_size, self.head, norm_mode=self.norm_mode, model=self.prehead,
                          first_layer=self.first_layer, backbone_input=self.backbone_input)

        s_time = time.time()
        model = self.optimize_params(trainer)
        repair_time = time.time() - s_time

        self.time_dict = {
            'repair': repair_time
        }

        model.eval()
        self.prehead = model
        self.prehead.cpu()
        self.head.eval()
        self.head.cpu()
        self.macs, self.params = PruneTrans.calculate_flops(self.prehead, input=torch.rand_like(self.x[:2]))

        self.macs_pre, self.params_pre = None, None
        if self.prehead.pre_resizer is not None:
            self.macs_pre, self.params_pre = PruneTrans.calculate_flops(self.prehead.pre_resizer,
                                                                        input=torch.rand_like(self.x[:2]))

    def optimize_params(self, trainer):
        # to device
        trainer = trainer.to(device)
        anchors = self.anchors.to(device)
        x = self.x.to(device)

        # loss function
        if 'npair' in self.loss_mode:
            # loss_fun = NPairLoss()
            # loss_fun = CrossSample(margin=np.max(self.margins), num_classes=self.num_classes,
            #             intra=np.min(self.intra))
            loss_fun = MMD()
        else:
            loss_fun = TripletLoss()

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01, momentum=0.95)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

        es_min = int(self.num_episode * 0.25)
        es = EarlyStopping(patience=10, min_break_epoch=es_min)

        best_loss = np.inf
        best_state_dict = None
        best_epoch = 0
        self.epoch_loss = []

        s_time = time.time()
        for t in range(self.num_episode):
            y_pred = trainer(x)
            # loss
            # loss = loss_fun(y_pred, self.y, anchors)
            loss = loss_fun(y_pred, self.y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()

            # update step
            optimizer.step()
            scheduler.step()

            # save epoch loss
            self.epoch_loss.append(loss.item())

            if es.step(loss):
                print('Early Stop>>>')
                print('Episode {:05} >>> Best loss is: {:.5f}'.format(t, best_loss))
                break

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state_dict = copy.deepcopy(trainer.model.state_dict())
                best_epoch = t

            # if t == 0 or t == self.num_episode - 1:
            if t%10 == 0:
                print('Episode {:05} >>> Loss is: {:.5f}'.format(t + 1, loss.item()))
        # Retrieve model
        print('Recover model weights on epoch #{}'.format(best_epoch))
        trainer.model.load_state_dict(best_state_dict)
        print('Total training time is {:.3f}'.format(time.time() - s_time))
        return trainer.model

    def get_prehead(self):
        return self.prehead.cpu()

    def get_macs_params(self):
        return self.macs, self.params

    def get_anchor_score(self):
        return self.anchor_score

    def get_time_dict(self):
        return self.time_dict

    def get_mac_params_pre(self):
        return self.macs_pre, self.params_pre

    def get_epoch_loss(self):
        return self.epoch_loss


class LastTrans:
    def __init__(self, x, y, num_episode=100, **kwargs):
        self.num_classes = len(torch.unique(y))
        self.x = x
        self.y = y
        self.num_episode = num_episode
        self.norm_mode = kwargs['norm_mode']
        self.update_param()

    def update_param(self):
        dim_in = self.x.size(1)
        input_size = self.x.size(2)
        dim_out = self.num_classes
        # trainer = TrainerCNN(dim_in, dim_out, norm_mode=self.norm_mode)
        # self.model = self.optimize_params(trainer)
        model = Conv(dim_in=dim_in, dim_out=dim_out, input_size=input_size)
        self.model = self.optimize_params(model)

    def optimize_params(self, trainer):
        # to device
        trainer = trainer.to(device)
        x = self.x.detach().to(device)
        y = self.y.to(device)

        # loss_fun = NPairLoss()
        loss_fun = nn.CrossEntropyLoss()
        # optimizer and scheduler
        # optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01, momentum=0.95)
        optimizer = torch.optim.SGD(trainer.parameters(), lr=0.01, momentum=0.95)
        scheduler = StepLR(optimizer, step_size=int(self.num_episode * 0.2), gamma=0.5)
        es_min = int(self.num_episode * 0.25)
        es = EarlyStopping(patience=10, min_break_epoch=es_min)
        s_time = time.time()
        # trainer.model.train()
        trainer.train()

        best_state_dict = None
        best_loss = np.inf
        best_epoch = 0

        for t in range(self.num_episode):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = trainer(x)

            # loss
            loss = loss_fun(y_pred, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()

            # update step
            optimizer.step()
            scheduler.step()

            if es.step(loss):
                print('Early Stop>>>')
                print('Episode {:05} >>> Best loss is: {:.5f}'.format(t, best_loss))
                break

            if loss.item() < best_loss:
                best_loss = loss.item()
                # best_state_dict = copy.deepcopy(trainer.model.state_dict())
                best_state_dict = copy.deepcopy(trainer.state_dict())
                best_epoch = t
            if t == 0 or t == self.num_episode - 1:
                print('Episode {:05} >>> Loss is: {:.5f}'.format(t + 1, loss.item()))
        print('Recover model weights on epoch #{}'.format(best_epoch))
        # trainer.model.load_state_dict(best_state_dict)
        trainer.load_state_dict(best_state_dict)
        print('Total training time is {:.3f}'.format(time.time() - s_time))
        # return trainer.model
        return trainer

    def get_model(self):
        return self.model.cpu()


class NormTrans:
    def __init__(self, x, y, anchor_x, anchor_y, test_x, test_y, head, num_episode=100, **kwargs):
        self.training = True
        self.num_classes = len(torch.unique(y))
        self.x = x
        self.y = y
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.test_x = test_x
        self.test_y = test_y
        self.head = head
        self.num_episode = num_episode
        self.norm_mode = kwargs['norm_mode']
        self.out_size = kwargs['out_size']
        self.update_param(x, y)

    def update_param(self, x, y):

        dim_in = self.x.size(1)
        dim_out = self.out_size[0]
        input_size = self.x.size(2)
        output_size = self.out_size[1]
        out_channels = self.anchor_x.size(1)
        trainer = TrainerNorm(dim_in=dim_in, dim_out=dim_out, input_size=input_size, output_size=output_size,
                              head=self.head, norm_mode=self.norm_mode, num_classes=self.num_classes,
                              out_channels=out_channels)

        model = self.optimize_params(trainer)
        self.prehead = model

        x = x.to(device)
        print('After repairing train S-score:')
        # Notes.write('After repairing train S-score:')
        self.score, _ = quick_check(x, y, self.anchor_x, self.anchor_y, head=self.head, model=model,
                                    norm_mode=self.norm_mode)
        print('After repairing test S-score:')
        # Notes.write('After repairing test S-score:')
        test_x = self.test_x.to(device)
        quick_check(test_x, self.test_y, self.anchor_x, self.anchor_y, head=self.head, model=model,
                    norm_mode=self.norm_mode)

    def optimize_params(self, trainer):
        # to device
        trainer = trainer.to(device)
        x = self.x.to(device)
        y = self.y.to(device).long()
        #### First Stage ####
        loss_fun = nn.CrossEntropyLoss()
        # optimizer and scheduler
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01, momentum=0.95)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
        s_time = time.time()
        # self.head.train()
        for t in range(self.num_episode):

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = trainer(x)

            # loss
            loss = loss_fun(y_pred, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()

            # update step
            optimizer.step()
            scheduler.step()
            if t == 0 or t == self.num_episode - 1:
                print('Episode {:05} >>> Loss is: {:.5f}'.format(t + 1, loss.item()))
        print('Total training time is {:.3f}'.format(time.time() - s_time))

        return trainer.model

    def get_prehead(self):
        return self.prehead.cpu()

    def get_afterhead(self):
        return None

    def get_score(self):
        return self.score


class RepairTrans:
    def __init__(self, x, y, anchor_x, anchor_y, head, num_episode=100, n_comp=2, **kwargs):
        self.training = True
        self.num_classes = len(torch.unique(y))
        self.x = x
        self.y = y
        self.head = head
        self.num_episode = num_episode
        self.num_episode_rot = kwargs['num_episode_rot']
        self.out_size = kwargs['out_size']
        self.n_comp = n_comp
        self.layer_id = kwargs['layer_id']
        self.mode = kwargs['mode']
        self.norm_mode = kwargs['norm_mode']
        self.regression = kwargs['regression']
        self.test_x = kwargs['test_x']
        self.test_y = kwargs['test_y']
        self.rotate = kwargs['rm']
        self.best_anchor_mode = kwargs['best_anchor']
        self.first_layer = kwargs['first_layer']
        self.backbone_input = kwargs['backbone_input']
        if 'prehead' in kwargs:
            self.prehead = kwargs['prehead']
        else:
            self.prehead = None
        print("Numpy sampling seed: {}".format(np.random.get_state()[1][0]))

        if self.regression:
            self.le = preprocessing.LabelEncoder()
            self.le.fit(self.y)
            self.y = self.le.transform(self.y)
        self.update_param(x, y, anchor_x, anchor_y)

    def update_param(self, x, y, anchor_x, anchor_y):
        start_time = time.time()
        ax_mmc = mmc(anchor_x)

        if 'I' in self.norm_mode:
            ax_mmc = F.normalize(ax_mmc)

        fitted_pca = pca_fit(ax_mmc, n_comp=self.n_comp, labels=anchor_y)
        self.pca = fitted_pca
        pca_time = time.time()

        # ax_pca = fitted_pca.transform(ax_mmc)
        # aScore = class_silhouette_score(ax_pca, anchor_y)
        # aTOPN = select_topN(aScore, top_n=self.num_classes)
        # mask = np.isin(anchor_y, aTOPN)
        # anchor_x = anchor_x[mask]
        # anchor_y = anchor_y[mask]
        # ax_mmc = mmc(anchor_x)
        # if 'I' in self.norm_mode:
        #     ax_mmc = F.normalize(ax_mmc)
        # fitted_pca = pca_fit(ax_mmc, n_comp=self.n_comp)
        if fitted_pca.__class__.__name__ == 'PCA':
            anchor_pca = torch.from_numpy(fitted_pca.components_.T).float()
            anchor_mean = torch.from_numpy(fitted_pca.mean_).float()
        elif fitted_pca.__class__.__name__ == 'SparsePCA':
            anchor_pca = torch.from_numpy(fitted_pca.components_.T).float()
            anchor_mean = torch.from_numpy(fitted_pca.mean_).float()
            # self.rotate = False
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=1)
            pca_fitted = pca.fit(ax_mmc)
            anchor_pca = torch.from_numpy(pca_fitted.components_.T).float()
            anchor_mean = torch.from_numpy(pca_fitted.mean_).float()
            # self.rotate = False

        # build trainer
        dim_in = self.x.size(1)
        dim_out = self.out_size[0]
        input_size = self.x.size(2)
        output_size = self.out_size[1]
        trainer = Trainer(anchor_pca, anchor_mean, dim_in, dim_out, input_size,
                          output_size, self.head, norm_mode=self.norm_mode, model=self.prehead,
                          first_layer=self.first_layer, backbone_input=self.backbone_input)

        # tx
        pre_x = trainer.model(x)
        pre_x = self.head(pre_x)

        pre_x_mmc = mmc(pre_x)
        if 'S' in self.norm_mode:
            pre_x_mmc = F.normalize(pre_x_mmc)
        pre_x_mmc = pre_x_mmc.detach().cpu()

        # pcas
        ax_pca = fitted_pca.transform(ax_mmc)
        tx_pca = fitted_pca.transform(pre_x_mmc)
        # add_to_dict('L{}_og'.format(self.layer_id), tx_pca)

        # ax properties
        aScore = class_silhouette_score(ax_pca, anchor_y, regression=self.regression)
        aTOPN = select_topN(aScore, top_n=self.num_classes)

        # scale
        tx_interD = inter_distance(tx_pca, y)

        # anchor margins
        if self.best_anchor_mode:
            best_ax_pca = []
            best_y = []
            for a in aTOPN:
                best_ax_pca.append(ax_pca[anchor_y == a])
                best_y.append(anchor_y[anchor_y == a])

            best_ax_pca = np.concatenate(best_ax_pca, axis=0)
            best_y = np.concatenate(best_y, axis=0)

            ax_intraD = intra_distance(best_ax_pca, best_y)
            ax_interD = inter_distance(best_ax_pca, best_y)
            ax_margins = ax_interD - ax_intraD
            scale = np.max(ax_interD, keepdims=True) / np.max(tx_interD, keepdims=True)

        else:
            ax_intraD = intra_distance(ax_pca, anchor_y)
            ax_interD = inter_distance(ax_pca, anchor_y)
            ax_margins = ax_interD - ax_intraD
            scale = np.mean(ax_interD[aTOPN], keepdims=True) / np.mean(tx_interD, keepdims=True)
        print('Scale is {:.4f}'.format(scale[0]))

        # maxI = np.argwhere(ax_interD == np.max(ax_interD))[0][0]
        # maxS = np.argwhere(tx_interD == np.max(tx_interD))[0][0]
        # print('Scale is {:.4f}, anchor class is {}, sensing class is {}'.format(scale[0], maxI, maxS))
        trainer.set_scale(torch.from_numpy(scale))
        tx_pca *= scale

        # get centroids for each class
        aMedoids = class_centroids(ax_pca, anchor_y)
        xMedoids = class_centroids(tx_pca, y)

        if self.rotate:
            sensing_anchor = np.mean(xMedoids, axis=0, keepdims=True)
            img_anchor = aMedoids[aTOPN, :]
            img_anchor = np.mean(img_anchor, axis=0, keepdims=True)

            trainer_matrix = Trainer_RotationMatrix(np.shape(sensing_anchor)[1], sensing_anchor, img_anchor,
                                                    layer_id=self.layer_id)
            trainer_matrix.train(epoch=self.num_episode_rot)

            rm = trainer_matrix.get_rm()
            tx_pca_r = np.matmul(tx_pca, rm.numpy())
            xMedoids_r = class_centroids(tx_pca_r, y)

        xScore = class_silhouette_score(tx_pca, y, regression=self.regression)
        xTOPN = select_topN(xScore, self.num_classes)

        self.anchor_score = np.mean([aScore[a] for a in aTOPN])

        class_map, dist = map_classes_best(aMedoids, aTOPN, xMedoids, xTOPN)
        if self.rotate:
            class_map_rm, dist_rm = map_classes_best(aMedoids, aTOPN, xMedoids_r, xTOPN)
            print('After rotation distance ({:.3f}), original distance ({:.3f}).'.format(dist_rm, dist))
            # set rm to forward function
            class_map = class_map_rm
            trainer.set_rm(rm)
            # add_to_dict('RM{}'.format(self.layer_id), rm.numpy())

        anchors = []
        anchor_ys = []
        margins = []
        intra = []
        inter = []
        for i in range(self.num_classes):
            anchors.append(aMedoids[class_map[i]])
            anchor_ys.append(class_map[i])
            if not self.best_anchor_mode:
                margins.append(ax_margins[class_map[i]])
                intra.append(ax_intraD[class_map[i]])
                inter.append(ax_interD[class_map[i]])
        anchors = np.stack(anchors)
        self.anchors = torch.from_numpy(anchors)

        if not self.best_anchor_mode:
            self.margins = np.stack(margins)
            self.intra = np.stack(intra)
            self.inter = np.stack(inter)
        else:
            self.margins = np.stack(ax_margins)
            self.intra = np.stack(ax_intraD)
            self.inter = np.stack(ax_interD)

        # for i in range(self.num_classes):
        #     anchors.append(aMedoids[alabels == class_map[i]])
        #     margins.append(ax_margins[alabels == class_map[i]])
        #     anchor_ys.append(class_map[i])
        #
        # anchors = np.concatenate(anchors, axis=0)
        # self.margins = np.concatenate(margins, axis=0)
        # self.anchors = torch.from_numpy(anchors)

        np_class_map = np.asarray([[a, b] for a, b in class_map.items()])
        # add_to_dict('C{}'.format(self.layer_id), np_class_map)
        # add_to_dict('A{}'.format(self.layer_id), self.anchors.numpy())
        # add_to_dict('Ay{}'.format(self.layer_id), anchor_ys)
        # add_to_dict('S{}'.format(self.layer_id), scale)

        matching_time = time.time()

        model = self.optimize_params(trainer)

        repair_time = time.time()

        self.time_dict = {
            'pca': pca_time - start_time,
            'matching': matching_time - pca_time,
            'repair': repair_time - matching_time
        }

        model.eval()
        self.prehead = model
        self.prehead.cpu()
        self.head.eval()
        self.head.cpu()
        self.macs, self.params = PruneTrans.calculate_flops(self.prehead, input=torch.rand_like(self.x[:2]))

        self.macs_pre, self.params_pre = None, None
        if self.prehead.pre_resizer is not None:
            self.macs_pre, self.params_pre = PruneTrans.calculate_flops(self.prehead.pre_resizer,
                                                                        input=torch.rand_like(self.x[:2]))

        # after
        pre_x = model(x)
        pre_x = self.head(pre_x)
        pre_x_mmc = mmc(pre_x)
        if 'S' in self.norm_mode:
            pre_x_mmc = F.normalize(pre_x_mmc)
        pre_x_mmc = pre_x_mmc.detach().cpu()

    def optimize_params(self, trainer):
        # to device
        trainer = trainer.to(device)
        # print(trainer.model)
        anchors = self.anchors.to(device)
        x = self.x.to(device)
        #### First Stage ####
        # loss function
        print('Max (inter-intra)_margin is {:.4f}, Min intra-margin is {:.4f}'.format(np.max(self.margins),
                                                                                      np.min(self.intra)))
        # print("Numpy sampling seed: {}".format(np.random.get_state()[1][0]))
        if self.regression:
            # loss_fun = CrossSample(margin=np.max(self.margins), num_classes=self.num_classes,
            #                        intra=np.min(self.intra), sampling_method='anchor_npair_regression')
            loss_fun = PositiveNegativeLoss(margin=np.max(self.margins), num_classes=self.num_classes,
                                            intra=np.min(self.intra))
        else:
            loss_fun = CrossSample(margin=np.max(self.margins), num_classes=self.num_classes,
                                   intra=np.min(self.intra))

            # loss_fun = MMD()




        # loss_fun = AnchorLoss()
        # loss_fun1 = NPairLoss()
        # optimizer and scheduler

        # bert model

        no_decay = ['bias', 'LayerNorm.weight']
        bias_params = [p for n, p in trainer.head.named_parameters() if not any(nd in n for nd in no_decay)]
        norm_params = [p for n, p in trainer.head.named_parameters() if any(nd in n for nd in no_decay)]
        bert_params = bias_params + norm_params

        # optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01, momentum=0.95)
        optimizer = torch.optim.SGD(list(trainer.model.parameters()) + list(bert_params), lr=0.01, momentum=0.95)
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
        scheduler = StepLR(optimizer, step_size=int(self.num_episode * 0.2), gamma=0.5)

        # early stop min
        es_min = int(self.num_episode * 0.25)
        es = EarlyStopping(patience=10, min_break_epoch=es_min)

        s_time = time.time()
        trainer.model.train()
        best_state_dict = None
        best_loss = np.inf
        best_epoch = 0
        self.epoch_loss = []
        # self.head.train()
        for t in range(self.num_episode):

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = trainer(x)

            # loss
            loss = loss_fun(y_pred, self.y, anchors)
            # loss1 = loss_fun1(y_pred, self.y)
            # loss = loss + loss1

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()

            # update step
            optimizer.step()
            scheduler.step()

            # save epoch loss
            self.epoch_loss.append(loss.item())

            if es.step(loss):
                print('Early Stop>>>')
                print('Episode {:05} >>> Best loss is: {:.5f}'.format(t, best_loss))
                break
            # print
            # if (t + 1) % 100 == 0 :
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state_dict = copy.deepcopy(trainer.model.state_dict())
                best_epoch = t

            if t == 0 or t == self.num_episode - 1:
                print('Episode {:05} >>> Loss is: {:.5f}'.format(t, loss.item()))
        # Retrieve model
        print('Recover model weights on epoch #{}'.format(best_epoch))
        trainer.model.load_state_dict(best_state_dict)
        print('Total training time is {:.3f}'.format(time.time() - s_time))
        # y_pred = trainer(x)
        # add_to_dict('L{}_new'.format(self.layer_id), y_pred.detach().cpu().numpy())
        return trainer.model

    def get_prehead(self):
        return self.prehead.cpu()

    def get_macs_params(self):
        return self.macs, self.params

    def get_anchor_score(self):
        return self.anchor_score

    def get_time_dict(self):
        return self.time_dict

    def get_mac_params_pre(self):
        return self.macs_pre, self.params_pre

    def get_epoch_loss(self):
        return self.epoch_loss


class PruneTrans:
    def __init__(self, x, y, anchor_x, anchor_y, anchor_inter, head, n_comp=2, max_range=1.0,
                 regression=False, **kwargs):
        super(PruneTrans, self).__init__()
        self.x = x.cpu()
        self.y = y.cpu()
        self.anchor_out = anchor_x
        self.anchor_y = anchor_y
        self.anchor_inter = anchor_inter
        self.head = head.cpu()
        self.act_dict = []
        self.n_comp = n_comp
        self.prune_mode = 'residual'
        self.norm_mode = kwargs['norm_mode']
        self.max_range = max_range
        self.regression = regression
        # self.mode = 'L2'
        self.mode = 'PCA'
        self.get_fitted_pca()

    def get_fitted_pca(self):
        anchor_mmc = mmc(self.anchor_out)
        if 'I' in self.norm_mode:
            anchor_mmc = F.normalize(anchor_mmc)
        self.fitted_pca = pca_fit(anchor_mmc, n_comp=self.n_comp, labels=self.anchor_y)

    def get_layer_pca_weights(self, id):
        anchor_mmc = self.act_dict[id]
        if 'I' in self.norm_mode:
            anchor_mmc = F.normalize(anchor_mmc)
        fitted_pca = pca_fit(anchor_mmc, n_comp=self.n_comp, labels=self.anchor_y)
        weights = np.abs(fitted_pca.components_).mean(0)
        return weights

    # resnet
    @torch.no_grad()
    def prune(self):
        self.pruned_head = copy.deepcopy(self.head)
        # step1: prepare all convolution layers
        conv_objs = (nn.Conv2d, nn.Conv1d)
        convs = PruneTrans.find_layers(conv_objs, self.pruned_head)

        # step2: get all original anchor mmc
        self.form_anchor_activation(convs)

        # step3: loop each conv to find best prune plan
        self.mask_hook_handles = {}
        downsample = None

        for lid, (name, conv) in enumerate(convs.items()):
            if 'downsample' in name:
                continue
            next_key = get_next_key(convs, name)
            if next_key and 'conv2' in name and 'downsample' in next_key:
                downsample = convs[next_key]

            temp_mask = torch.ones(conv.out_channels)

            # add mask buffer
            conv.register_buffer('mask', temp_mask[None, :, None, None])
            if downsample:
                downsample.register_buffer('mask', temp_mask[None, :, None, None])

            # register hook to modify output
            handle = self.register_mask_hook(conv)
            self.mask_hook_handles[name] = handle
            if downsample:
                handle = self.register_mask_hook(downsample)
                self.mask_hook_handles[next_key] = handle

            # weight pca
            weights = self.get_layer_pca_weights(lid)
            sorts = np.argsort(weights)

            score_dic = {}

            if isinstance(conv, nn.Conv1d):
                self.is_1d = True
            else:
                self.is_1d = False

            for n in range(int(self.max_range * len(weights)) - 1):
                temp_mask = torch.ones(conv.out_channels)
                temp_mask[sorts[:n]] = 0
                conv.mask = self.expend_dim(temp_mask)
                if downsample:
                    downsample.mask = self.expend_dim(temp_mask)

                # feed x to layer with masked hook
                temp_x = self.pruned_head(self.x)
                temp_x *= conv.mask

                x_mmc = mmc(temp_x)
                if 'S' in self.norm_mode:
                    x_mmc = F.normalize(x_mmc)
                x_pca = self.fitted_pca.transform(x_mmc)

                sMap = class_silhouette_score(x_pca, self.y, regression=self.regression)
                score = np.mean([s for _, s in sMap.items()])
                score_dic[n] = np.round(score, 2)
            # print(score_dic)
            if int(self.max_range * len(weights)) - 1 <= 0:
                topn = 0
            else:
                topn = sorted(score_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]
                print('Best topK in layer {} is to close #{} channels resulting in S-score={:.4f}'.format(name, topn,
                                                                                                          score_dic[
                                                                                                              topn]))
                Notes.write('Pruning: Close #{} channels, S-score={:.4f}'.format(topn, score_dic[topn]))

            if self.mode == 'L2':
                copy_conv = copy.deepcopy(conv)
                prune.ln_structured(copy_conv, name='weight', amount=topn, dim=0, n=2)

                temp_mask = torch.mean(copy_conv.weight_mask, dim=(1, 2, 3))
                conv.mask = self.expend_dim(temp_mask)
                if downsample:
                    downsample.mask = self.expend_dim(temp_mask)
            else:
                temp_mask = torch.ones(conv.out_channels)
                temp_mask[sorts[:topn]] = 0
                conv.mask = self.expend_dim(temp_mask)
                if downsample:
                    downsample.mask = self.expend_dim(temp_mask)

            downsample = None

        self.prune_resnet()

    def expend_dim(self, tensor):
        if self.is_1d:
            return tensor[None, :, None]
        else:
            return tensor[None, :, None, None]

    def form_anchor_activation(self, convs):
        self.act_dict.insert(0, self.anchor_out)
        if len(convs) > 1:
            self.act_dict.insert(0, self.anchor_inter)

    def register_mask_hook(self, layer):
        handle = layer.register_forward_hook(get_mask_hook(layer))
        return handle

    def register_activation_hook(self, name, layer):
        handle = layer.register_forward_hook(get_activation_hook(name, self.act_dict))
        return handle

    def remove_hooks(self, handles):
        for handle in handles:
            handle.remove()

    def remove_mask_hooks(self):
        for key, handle in self.mask_hook_handles.items():
            handle.remove()

    @staticmethod
    def find_layers(objs, model):
        layers = {}
        for key, item in model.named_modules():
            if isinstance(item, objs):
                layers[key] = item
        return layers

    def prune_model(self):
        objs = (nn.Conv2d, nn.BatchNorm2d)
        layers = PruneTrans.find_layers(objs, self.pruned_head)

        pre_mask = None
        for name, layer in layers.items():
            if isinstance(layer, nn.Conv2d):
                mask = layer.mask.cpu().numpy().squeeze()
                if len(mask) > 0:
                    idxs = [id for id in np.argwhere(mask == 0).squeeze()]
                else:
                    idxs = []

                tp.prune_conv_out_channel(layer, idxs)
                if pre_mask:
                    tp.prune_conv_in_channel(layer, pre_mask)
                pre_mask = idxs

                # delete layer mask
                del layer.mask

            elif isinstance(layer, nn.BatchNorm2d):
                if pre_mask is not None:
                    tp.prune_batchnorm(layer, pre_mask)

        _remove_all_forward_hooks(self.pruned_head)
        input = torch.rand_like(self.x[:2])
        print('Before pruning >>>')
        PruneTrans.calculate_flops(self.head, input)
        print('After pruning >>>')
        PruneTrans.calculate_flops(self.pruned_head, input)

    def prune_resnet(self):
        input = torch.rand_like(self.x[:2])
        print('Before pruning >>>')
        PruneTrans.calculate_flops(self.head, input)

        model_dict = build_model_dict(self.pruned_head)
        objs = (nn.Conv2d, nn.Conv1d, nn.BatchNorm2d, nn.BatchNorm1d)
        conv_objs = (nn.Conv2d, nn.Conv1d)
        bn_objs = (nn.BatchNorm2d, nn.BatchNorm1d)

        layers = PruneTrans.find_layers(objs, self.pruned_head)

        pre_mask = None
        downsample = None
        block = None
        non_idxs = None
        for name, layer in layers.items():

            # convolution layer
            if isinstance(layer, conv_objs):
                if 'downsample' in name:
                    continue

                next_key = get_next_key(layers, name, skip_num=2)
                if next_key and 'conv2' in name and 'downsample' in next_key:
                    downsample = layers[next_key]
                elif 'conv2' in name:
                    b_name = name.replace('.conv2', '')
                    block = model_dict[b_name]
                    assert isinstance(block, (BasicBlock, BasicBlock1d)), "block must be BasicBlock object!"

                mask = layer.mask.cpu().numpy().squeeze()

                idxs = [id for id in np.argwhere(mask == 0).squeeze(axis=1)]
                non_idxs = [id for id in np.argwhere(mask == 1).squeeze(axis=1)]

                # prune output
                tp.prune_conv_out_channel(layer, idxs)
                if downsample:
                    tp.prune_conv_out_channel(downsample, idxs)
                if block:
                    block.downsample = Downsample(torch.from_numpy(np.asarray(non_idxs)))

                # prune input
                if pre_mask:
                    tp.prune_conv_in_channel(layer, pre_mask)

                pre_mask = idxs

                # delete layer mask
                del layer.mask
                if downsample:
                    del downsample.mask
                downsample = None
                block = None

            elif isinstance(layer, bn_objs):
                # pass
                if pre_mask is not None:
                    tp.prune_batchnorm(layer, pre_mask)

        self.masks = non_idxs

        _remove_all_forward_hooks(self.pruned_head)
        print('After pruning >>>')
        macs, params = PruneTrans.calculate_flops(self.pruned_head, input)
        self.macs = macs
        self.params = params
        # Notes.write('MACs is {}, Params is {}'.format(macs, params))

    @staticmethod
    def calculate_flops(model, input):
        macs, params = profile(model, inputs=(input,), verbose=False)
        new_macs, new_params = clever_format([macs, params], "%.3f")
        print('MACs is {}, Params is {}'.format(new_macs, new_params))
        return macs, params

    def get_macs_params(self):
        return self.macs, self.params

    def get_pruned_head(self):
        return self.pruned_head

    def get_masks(self):
        return self.masks


class Finetuner:
    def __init__(self, x, y, model, step, backnet, regression, **kwargs):
        super(Finetuner, self).__init__()
        self.x = x
        self.y = y
        self.model = model
        self.is_1d = kwargs['is_1d']
        y_pred = self.model(self.x)
        num_classes = len(np.unique(y.numpy()))
        num_channel = y_pred.size(1)
        self.steps = step
        self.regression = regression
        if regression:
            self.linear = Linear(num_channel, 1, self.is_1d)
            self.linear.linear.bias.data[0] = torch.max(y).item()
        else:
            self.linear = Linear(num_channel, num_classes,
                                 self.is_1d)
        self.backnet = backnet
        self.logger = kwargs['logger']

    def optimize_params(self):
        # to device
        self.model = self.model.train()
        model = self.model.to(device)
        x = self.x.to(device)
        y = self.y.to(device)
        self.linear = self.linear.to(device)

        # loss function
        if self.regression:
            loss_fun = nn.MSELoss()
        else:
            loss_fun = nn.CrossEntropyLoss()
        self.freeze_encoder()

        optimizer = torch.optim.SGD(list(model.parameters()) + list(self.linear.parameters()), lr=0.01, momentum=0.95)

        for t in range(self.steps):
            y_pred = model(x)
            y_pred = self.linear(y_pred)
            loss = loss_fun(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()

            # update step
            optimizer.step()
            if (t + 1) % 10 == 0:
                print("Step #{}".format(t + 1))
                model.cpu()
                model.eval()
                self.linear.cpu()
                self.linear.eval()
                self.test()
                model.to(device)
                model.train()
                self.linear.to(device)
                self.linear.train()
            if t == 0 or t == self.steps - 1:
                print('Episode {:05} >>> Loss is: {:.5f}'.format(t + 1, loss.item()))
        self.linear.cpu()
        self.linear.eval()

    def freeze_encoder(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, AutoEncoderOG):
                for param in layer.parameters():
                    param.requires_grad = False

    def get_linear(self):
        return self.linear

    def test(self):

        if not self.regression:
            print('Linear:')
            linear_train_acc, linear_test_acc = self.backnet.scorer_classifier(trained=True, linear=self.linear,
                                                                               og=True)
            self.logger.write('finetune_linear_train_acc', linear_train_acc)
            self.logger.write('finetune_linear_test_acc', linear_test_acc)

            print('KNN:')
            knn_train_acc, knn_test_acc = self.backnet.scorer()
            self.logger.write('finetune_knn_train_acc', knn_train_acc)
            self.logger.write('finetune_knn_test_acc', knn_test_acc)
        else:
            print('SVM Regression:')
            knn_train_acc, knn_test_acc = self.backnet.scorer_regression(method='SVR')
            self.logger.write('finetune_knn_train_acc', knn_train_acc)
            self.logger.write('finetune_knn_test_acc', knn_test_acc)

            print('Linear Regression:')
            linear_train_acc, linear_test_acc = self.backnet.scorer_regression(method='linear', trained=True, og=True,
                                                                               linear=self.linear)
            self.logger.write('finetune_linear_train_acc', linear_train_acc)
            self.logger.write('finetune_linear_test_acc', linear_test_acc)


if __name__ == "__main__":
    # net = Downsample(torch.from_numpy(np.asarray([0, 1, 2, 3, 4])))
    # input = torch.randn(3, 7, 4, 4)
    # out = net(input)
    # print(out.shape)

    from torchvision.models import resnet18

    net = resnet18()
    PruneTrans.calculate_flops(net, input=torch.rand(2, 3, 224, 224))

