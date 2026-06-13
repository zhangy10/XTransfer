# import torch
# import numpy as np
# import math
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Conv2d, ReLU, BatchNorm2d, MaxPool2d
from numpy.linalg import norm
import os
import copy
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import paired_distances
from collections import OrderedDict
import torchvision.transforms as transforms
import json

from thop import clever_format

from utils import load_pretrained_weights, read_json, load_dict, Notes, set_random_seed
from xtransfer.model_builder import BaseNet
from xtransfer.trans import PruneTrans, ResizeTrans, quick_check, NormTrans, bn_init, prototype_test, LastTrans, func, \
    fit_curve, calculate_anchor_score, MMCTrans
from xtransfer.trans import RepairTrans
from xtransfer.tools import class_accuracy, build_model_dict, get_next_key, class_centroids, class_silhouette_score, \
    inter_distance, mmc, convert_numpy
from xtransfer.hook import add_to_dict
from xtransfer.engine import SimpleTrainer, OXiodTrainer, OXiodLinear
# from dataloader.oxiod_dataset import load_oxiod_dataset, load_dataset_6d_quat, generate_trajectory_6d_quat
from dataloader.transforms import *
from modeling.backbone.conv1d import Conv4


class Seq(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        for module in self:
            input = module(input)
            if isinstance(input, tuple):
                input = input[0]
        return input


class SeqBase(nn.Module):
    def __init__(self):
        super(SeqBase, self).__init__()
        self.parametrized_layers = []
        self.trunk = None

    def build_layers(self):
        self.trunk = Seq(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        # for layer in self.parametrized_layers:
        #     x = layer(x)
        #     if isinstance(x, tuple):
        #         x = x[0]
        return out

    def add_layer(self, layer):
        if layer is not None:
            self.parametrized_layers.append(layer)
            self.build_layers()

    def remove_last_layer(self):
        self.parametrized_layers = self.parametrized_layers[:-1]
        self.build_layers()

    def add_layers(self, layers):
        self.parametrized_layers.extend(layers)
        self.build_layers()

    def clear(self):
        self.parametrized_layers = []
        self.build_layers()


class HeadTail:
    def __init__(self):
        self.clear_dict()

    def add_head(self, item):
        self.ht['head'] = item

    def add_tail(self, item):
        self.ht['tail'] = item
        return self.ht

    def check_head(self):
        return self.ht['head'] == None

    def get_dict(self):
        return self.ht

    def clear_dict(self):
        self.ht = dict(input_size=None, head=None, tail=None, anchor=None)


class MatchingNet:
    nnlayers = (Conv2d, ReLU, BatchNorm2d, MaxPool2d)

    def __init__(self, cfg, trainloader=None, testloader=None, valloader=None, fit_schema=None, logger=None, **kwargs):
        self.cfg = cfg
        self.pool = {}
        self.backbone = SeqBase()
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.fit_schema = fit_schema
        self.match_info = read_json(cfg.MODEL_POOL.MATCH_JSON)
        self.struc_info = read_json(cfg.MODEL_POOL.STRUC_JSON)
        self.named_struc_info = read_json(cfg.MODEL_POOL.NAMED_STRUC_JSON)
        self.n_shot = cfg.DATALOADER.NUM_SHOTS
        self.break_layer_id = cfg.RSR.BREAK_DEPTH
        self.test_acc = cfg.RSR.TEST_ACC
        self.test_method = cfg.RSR.TEST_METHOD
        self.norm_mode = cfg.RSR.NORM_MODE
        self.regression = cfg.RSR.REGRESSION
        self.n_comp = cfg.RSR.PCA_COMPONENT
        self.search_depth = cfg.RSR.SEARCH_DEPTH
        self.logger = logger
        self.seed_pool = OrderedDict()
        self.init_seed = None
        self.estimate_score_history = {}
        self.score_history = {}
        self.pick_range = cfg.RSR.INIT_PICK_RANGE
        self.break_switch = False
        self.backbone_history = []
        self.build_model_pool(cfg)
        self.split_model_to_layers_name()
        self.generate_layer_pools()

    def build_model_pool(self, cfg):

        model_name = cfg.MODEL_POOL.MODEL_NAME
        anchor_name = cfg.MODEL_POOL.ANCHOR_NAME
        model_dir = cfg.MODEL_POOL.DIR
        model_pools = cfg.MODEL_POOL.NAMES

        # load model path
        for key in model_pools:
            self.pool[key] = {}
            self.pool[key]['path'] = os.path.join(model_dir, key, model_name)

        # load models
        for key in self.pool.keys():
            self.pool[key]['model'] = self.create_load_model(key)
            # self.pool[key]['anchor'] = self.load_anchor(os.path.join(model_dir, key, anchor_name))

            anchor = self.load_anchor(os.path.join(model_dir, key, anchor_name))
            anchor = self.align_anchor_to_target(anchor)
            self.pool[key]['anchor'] = anchor

            self.pool[key]['label'] = self.pool[key]['anchor']['label']
            self.pool[key]['backbone'] = self.match_info[key]['backbone']

    def create_load_model(self, name):
        if 'Conv1d' in self.match_info[name]['backbone']:
            net = Conv4(self.match_info[name]['num_channels'])
        else:
            net = BaseNet(self.match_info[name]['backbone'], self.match_info[name]['num_classes'],
                          in_channels=self.match_info[name]['num_channels'])

            if hasattr(net, 'backbone'):
                net = net.backbone

            backbone = self.match_info[name]['backbone']
            width = self.named_struc_info[backbone]['input']['0'][1]

            self.named_struc_info[backbone]['input']['0'] = [self.match_info[name]['num_channels'], width, width]
        load_pretrained_weights(net, self.pool[name]['path'])
        net.eval()
        return net

    def load_anchor(self, path):
        if os.path.exists(path):
            anchor_dict = load_dict(path)
            for key, item in anchor_dict.items():
                if key != 'label':
                    anchor_dict[key] = torch.from_numpy(anchor_dict[key]).float()
                else:
                    anchor_dict[key] = torch.from_numpy(anchor_dict[key]).long()
            return anchor_dict
        else:
            print('There is no anchor activation!!!')
            return None

    def align_anchor_to_target(self, dict):
        aligned_anchor = {}
        label = dict['label'].numpy()
        class_dict = {}
        for i in np.unique(label):
            idx = np.argwhere(label == i).squeeze()
            class_dict[i] = idx

        for key, item in dict.items():
            item = item.numpy()
            all_means = []
            for cid, idx in class_dict.items():
                means = np.stack([np.mean(d, axis=0) for d in np.array_split(item[idx], self.n_shot)])
                all_means.append(means)
            if key == 'label':
                aligned_anchor[key] = torch.from_numpy(np.concatenate(all_means, axis=0)).long()
            else:
                aligned_anchor[key] = torch.from_numpy(np.concatenate(all_means, axis=0))
        return aligned_anchor

    def split_model_to_layers_index(self):
        sidx = [2, 4, 6, 7, 9, 10]
        for key in self.pool.keys():
            self.pool[key]['layers'] = {}
            m = self.pool[key]['model'].backbone
            backbone_name = self.pool[key]['model'].model_name
            i = 0
            uidx = 0
            inidx = 0
            layers = []
            ht = HeadTail()
            for name, l in m.named_modules():
                if isinstance(l, self.nnlayers):
                    if i in sidx:
                        block = SeqBase()
                        block.add_layers(layers)
                        if ht.check_head():
                            ht.add_head(block)
                        else:
                            unit = ht.add_tail(block)
                            unit['input_size'] = self.struc_info[backbone_name][str(inidx)]
                            ht.clear_dict()
                            self.pool[key]['layers'][uidx] = unit
                            inidx = i
                            uidx += 1
                        layers = []
                    nl = copy.deepcopy(l)
                    layers.append(nl)
                    i += 1
                    # print(l)
            # print()

    def split_model_to_layers_name(self):
        self.backbone_depth = 0
        for key in self.pool.keys():
            self.pool[key]['layers'] = {}
            backbone_name = self.pool[key]['backbone']
            m = self.pool[key]['model']
            names = self.named_struc_info[backbone_name]['struc']
            depth = len(list(names))
            if depth > self.backbone_depth:
                self.backbone_depth = depth
            mdict = build_model_dict(m)
            for lid, item in names.items():
                ht = HeadTail()
                # add head
                hd = SeqBase()
                if len(item['head']) > 0:
                    for lname in item['head']:
                        hd.add_layer(mdict[lname])
                    ht.add_head(hd)
                else:
                    ht.add_head(None)

                # add tail
                tl = SeqBase()
                if len(item['tail']) > 0:
                    for lname in item['tail']:
                        tl.add_layer(mdict[lname])
                    unit = ht.add_tail(tl)

                else:
                    unit = ht.add_tail(None)

                # add anchor
                if len(item['head']) > 0:
                    out_key = '{}_out'.format(item['head'][-1])
                    unit['anchor'] = self.pool[key]['anchor'][out_key]

                    inter_key = self.search_inter_key(self.pool[key]['anchor'], item['head'][-1])
                    if inter_key:
                        unit['anchor_inter'] = self.pool[key]['anchor'][inter_key]
                    else:
                        unit['anchor_inter'] = None

                    unit['anchor_label'] = self.pool[key]['label']

                # input size
                unit['input_size'] = tuple(self.named_struc_info[backbone_name]['input'][str(lid)])
                # add to layer pool
                ht.clear_dict()
                self.pool[key]['layers'][lid] = unit
                # print()

    def search_inter_key(self, dic, block_key):
        for key in dic.keys():
            if block_key == key[:len(block_key)] and key != (block_key + '_out'):
                # print(key)
                return key
        return None

    def generate_layer_pools(self):
        self.layer_pool = defaultdict(defaultdict)

        for key in self.pool.keys():
            for lid in self.pool[key]['layers'].keys():
                self.layer_pool[lid]['{}_{}'.format(key, lid)] = self.pool[key]['layers'][lid]

    def fit(self):
        # prepare datasets
        train_iter = iter(self.trainloader)
        test_iter = iter(self.testloader)
        val_iter = iter(self.valloader)
        train_x, train_y = next(train_iter)
        test_x, test_y = next(test_iter)
        val_x, val_y = next(val_iter)

        og_input = copy.deepcopy(train_x)
        og_test = copy.deepcopy(test_x)
        og_val = copy.deepcopy(val_x)

        # init rate data
        self.rate_data = {'x': [0, self.backbone_depth], 'y': [0, 2]}

        add_to_dict('label', train_y)
        add_to_dict('label_test', test_y)

        # init params
        self.backbone_layer = 1
        last_lid = -1
        self.last_train_score = -1
        self.last_test_score = -1
        self.last_val_score = -1
        self.total_macs = 0
        self.total_params = 0

        # scorer prepare
        if self.regression:
            scorer = self.scorer_regression
        else:
            scorer = self.scorer

        for lid in self.layer_pool.keys():
            pool_time_dict = {}
            search_depth = 1 if int(lid) == 0 else self.search_depth

            # break
            if int(lid) > self.break_layer_id:
                print('COMPLETE!')
                break
            # if self.backbone_layer > 2:
            #     print('COMPLETE!')
            #     break

            if self.cfg.RSR.REGRESSION and self.backbone_layer > 2:
                print('COMPLETE!')
                break

            # if int(lid) in [1, 2, 3, 4, 5, 6]:
            #     print('SKIP!')
            #     continue
            # if int(lid) == 7:
            # self.cfg.RSR.LOSS_MODE = 'mmc-triplet'
            # self.cfg.RSR.LOSS_MODE = 'mmc-npair'

            # skip already selected lid
            if int(lid) <= int(last_lid):
                print('Current #{}, Last selected #{} >>> Skip.'.format(lid, last_lid))
                continue

            print('Search for backbone layer {}:'.format(self.backbone_layer))
            Notes.write('Search for backbone layer {}:'.format(self.backbone_layer))

            # Selecting layers
            layer_candidates = defaultdict(defaultdict)
            layer_pool = defaultdict()
            for i in range(search_depth):
                next_id = get_next_key(self.layer_pool, lid, skip_num=i)
                if next_id is not None and int(next_id) <= self.break_layer_id:
                    lpool = self.layer_pool[next_id]
                    layer_pool.update(lpool)

            # quick check
            if self.cfg.RSR.QUICK_CHECK and int(lid) != 0:
                stime = time.time()
                layer_pool, layers_score = self.check_score(layer_pool, train_x, train_y, test_x=test_x, test_y=test_y,
                                                            val_x=val_x,
                                                            val_y=val_y)
                
                # Get the layer with highest estimate_score
                # top_layer = max(layers_score.keys(), key=lambda k: layers_score[k]['estimated_score'])
                # layer_pool = {top_layer: layer_pool[top_layer]}

                quickcheck_time = time.time() - stime
                pool_time_dict['quick_check'] = quickcheck_time

            for nid, dictL in layer_pool.items():
                if self.cfg.RSR.POOL_SEED:
                    if self.backbone_layer not in self.seed_pool.keys():
                        self.seed_pool[self.backbone_layer] = np.random.get_state()
                    np.random.set_state(self.seed_pool[self.backbone_layer])
                else:
                    # set_random_seed(5)
                    if self.init_seed is None:
                        self.init_seed = np.random.get_state()
                    np.random.set_state(self.init_seed)

                # encoder&decoder build
                x = int(nid.split('_')[-1])
                layer_dict = self.select_layer(dictL, train_x, train_y, lid=x, nid=nid, test_x=test_x,
                                               test_y=test_y, val_x=val_x, val_y=val_y)
                layer_candidates[nid] = layer_dict
                pool_time_dict[nid] = layer_dict['time']

            # select best layers according to score
            best_nid = self.select_best_layer(layer_candidates)

            # for experiment only
            # best_nid = list(layer_pool.keys())[0]

            self.logger.write('layer_time', (best_nid, pool_time_dict))
            self.print_dict(layer_candidates)

            if best_nid is None:
                last_lid = int(last_lid) + search_depth
                print('Could not find better layer in #{} layer pool'.format(lid))
                Notes.write('Could not find better layer in #{} layer pool'.format(lid))
                continue

            if self.break_switch:
                print("Reach the maximum resource capacity, Complete")
                break

            # logger
            if self.cfg.RSR.QUICK_CHECK and int(lid) != 0:
                self.logger.write('estimate_score', layers_score[best_nid]['estimated_score'])
            self.logger.write('score', layer_candidates[best_nid]['train_score'])
            self.logger.write('epoch_loss', layer_candidates[best_nid]['epoch_loss'])

            # update rate data
            self.backbone_history.append(best_nid)
            self.logger.write('backbone_structure', "->".join(self.backbone_history))
            self.rate_data['x'].append(self.backbone_layer)
            self.rate_data['y'].append(layer_candidates[best_nid]['train_rate'])
            self.score_history[self.backbone_layer] = round(layer_candidates[best_nid]['train_score'], 4)

            last_lid = int(best_nid.split('_')[-1])
            print('Selected best layer {}'.format(best_nid))
            Notes.write('Selected best layer {}'.format(best_nid))
            # self.last_train_score = layer_candidates[best_nid]['prune_train_score']
            # self.last_test_score = layer_candidates[best_nid]['prune_test_score']
            # self.last_val_score = layer_candidates[best_nid]['prune_val_score']

            self.last_train_score = layer_candidates[best_nid]['train_score']
            self.last_test_score = layer_candidates[best_nid]['test_score']
            self.last_val_score = layer_candidates[best_nid]['val_score']

            print('Current Best Train score:{} , Test score:{}, Val score:{}'.format(self.last_train_score,
                                                                                     self.last_test_score,
                                                                                     self.last_val_score))
            Notes.write('Current Best Train score:{} , Test score:{}, Val score:{}'.format(self.last_train_score,
                                                                                           self.last_test_score,
                                                                                           self.last_val_score))
            print("Select pretrained weight from {} for backbone layer #{}".format(best_nid, int(self.backbone_layer)))

            # set pca
            self.pca = layer_candidates[best_nid]['pca_fun']

            # add resizer
            self.backbone.add_layer(layer_candidates[best_nid]['resizer'])

            # add og and test, finally delete layers
            self.backbone.add_layer(layer_candidates[best_nid]['prehead_og'])
            self.backbone.add_layer(layer_candidates[best_nid]['head_og'])

            if self.test_acc:
                print('Before repairing >>>')
                if not self.regression:
                    self.check_metric()
                train_acc, test_acc = scorer(method=self.test_method)
                print('PCA TEST>>>')
                train_pca_acc, test_pca_acc = scorer(method=(self.test_method + '_pca'))
                self.logger.write('before_train_acc', train_acc)
                self.logger.write('before_test_acc', test_acc)
                self.logger.write('before_train_pca_acc', train_pca_acc)
                self.logger.write('before_test_pca_acc', test_pca_acc)

            input = self.backbone(og_input)
            test = self.backbone(og_test)
            # add_to_dict('L{}og'.format(lid), input.detach().cpu().numpy())
            # add_to_dict('L{}og_test'.format(lid), test.detach().cpu().numpy())

            self.backbone.remove_last_layer()
            self.backbone.remove_last_layer()

            # add prehead
            self.backbone.add_layer(layer_candidates[best_nid]['prehead'])

            # add head
            self.backbone.add_layer(layer_candidates[best_nid]['head'])
            input = self.backbone(og_input)
            test = self.backbone(og_test)
            # add_to_dict('L{}'.format(lid), input.detach().cpu().numpy())
            # add_to_dict('L{}_test'.format(lid), test.detach().cpu().numpy())

            if self.test_acc:
                print('After repairing S2>>>')
                if not self.regression:
                    self.check_metric()
                train_acc, test_acc = scorer(method=self.test_method)
                print('PCA TEST>>>')
                train_pca_acc, test_pca_acc = scorer(method=(self.test_method + '_pca'))
                self.logger.write('train_acc', train_acc)
                self.logger.write('test_acc', test_acc)
                self.logger.write('train_pca_acc', train_pca_acc)
                self.logger.write('test_pca_acc', test_pca_acc)
                self.logger.write('train_pt_acc', layer_candidates[best_nid]['train_acc'])
                self.logger.write('test_pt_acc', layer_candidates[best_nid]['test_acc'])

            # remove pretrained layer and add pruned pretrained layer
            if 'pruned_head' in layer_candidates[best_nid]:
                self.backbone.remove_last_layer()
                self.backbone.add_layer(layer_candidates[best_nid]['pruned_head'])
                input_after = self.backbone(og_input)
                # add_to_dict('L{}_afterhead'.format(lid), input_after.detach().cpu().numpy())
                if self.test_acc:
                    print('After pruning >>>')
                    train_acc, test_acc = scorer(method=self.test_method)
                    print('PCA TEST>>>')
                    train_pca_acc, test_pca_acc = scorer(method=(self.test_method + '_pca'),
                                                         prune_mask=layer_candidates[best_nid]['prune_masks'])

                    self.logger.write('prune_train_acc', train_acc)
                    self.logger.write('prune_test_acc', test_acc)
                    self.logger.write('prune_train_pca_acc', train_pca_acc)
                    self.logger.write('prune_test_pca_acc', test_pca_acc)
                    self.logger.write('prune_train_pt_acc', layer_candidates[best_nid]['prune_train_acc'])
                    self.logger.write('prune_test_pt_acc', layer_candidates[best_nid]['prune_test_acc'])

            # add tail
            if 'tail' in layer_candidates[best_nid]:
                self.backbone.add_layer(layer_candidates[best_nid]['tail'])

            # update input
            self.backbone.eval()

            train_x = self.backbone(og_input)
            test_x = self.backbone(og_test)
            val_x = self.backbone(og_val)
            self.fdim_out = train_x.size(1)

            self.total_macs += (layer_candidates[best_nid]['macs'] + layer_candidates[best_nid]['encoder_macs'])
            self.total_params += (layer_candidates[best_nid]['params'] + layer_candidates[best_nid]['encoder_params'])
            self.logger.write('macs', self.total_macs)
            self.logger.write('params', self.total_params)
            self.logger.write('params', self.total_params)

            # sequence backbone layer id
            self.backbone_layer += 1

        # end add conv layer to
        if self.cfg.RSR.END_CNN:
            print('Add last CNN model >>>')
            cnn_trans = self.build_transformer('cnn', input=train_x, label=train_y, anchor_input=None,
                                               anchor_label=None)
            model = cnn_trans.get_model()
            model.eval()
            self.backbone.add_layer(model)
            print('After Last CNN >>>')
            Notes.write('After Last CNN >>>')
            self.scorer_classifier(trained=True)

        # end print resource
        macs, params = clever_format([self.total_macs, self.total_params], "%.2f")
        print('Total MACs is {}, Total params is {}'.format(macs, params))
        Notes.write('Total MACs is {}, Total params is {}'.format(macs, params))

    def print_dict(self, layer_candidates):
        scores = {k: {'before_train_score': round(v['before_train_score'], 4),
                      'bn_train_score': round(v['bn_train_score'], 4),
                      'train_score': round(v['train_score'], 4),
                      'prune_train_score': round(v['prune_train_score'], 4),

                      'before_test_score': round(v['before_test_score'], 4),
                      'bn_test_score': round(v['bn_test_score'], 4),
                      'test_score': round(v['test_score'], 4),
                      'prune_test_score': round(v['prune_test_score'], 4),

                      'before_val_score': round(v['before_val_score'], 4),
                      'bn_val_score': round(v['bn_val_score'], 4),
                      'val_score': round(v['val_score'], 4),
                      'prune_val_score': round(v['prune_val_score'], 4),

                      'grad_score': round(v['grad_score'], 4),
                      'anchor_score': round(v['anchor_score'], 4),
                      'improved_train_score': round(v['train_score'] - v['before_train_score'], 4),
                      'improved_test_score': round(v['test_score'] - v['before_test_score'], 4),
                      'improved_val_score': round(v['val_score'] - v['before_val_score'], 4),
                      'train_rate': round(v['train_rate'], 3),
                      'test_rate': round(v['test_rate'], 3),

                      'train_acc': round(v['train_acc'], 4),
                      'test_acc': round(v['test_acc'], 4),
                      'prune_train_acc': round(v['prune_train_acc'], 4),
                      'prune_test_acc': round(v['prune_test_acc'], 4),

                      'total_macs': v['all_macs_p'], 'total_params': v['all_params_p'],
                      'pre_macs': v['pre_macs_p'], 'pre_params': v['pre_params_p'],
                      'encoder_macs': v['encoder_macs_p'], 'encoder_params': v['encoder_params_p'],

                      'layer_macs': v['macs_p'], 'layer_params': v['params_p'],
                      's-score': round(v['s_score'], 4),
                      'resource_score': round(v['resource_score'], 4),
                      'score': round(v['score'], 4),
                      } for k, v in layer_candidates.items()}
        scores = convert_numpy(scores)
        scores = json.dumps(scores, indent=4)
        print(scores)
        Notes.write(scores)
        print()

    def select_best_layer(self, layer_candidates):
        # calculate score
        print('Current search layer:{}, backbone depth:{}'.format(self.backbone_layer, self.backbone_depth))
        layer_max = np.exp(self.backbone_layer / (self.backbone_depth / 2) - 2) + 1
        floats_max = max([item['encoder_macs'] + item['macs'] for item in layer_candidates.values()])
        floats_scale = floats_max / layer_max
        if self.cfg.RSR.BEST_VALIDATION:
            a = 0.5
            b = 0.5
        else:
            a = 1
            b = 0

        for key, item in layer_candidates.items():
            if item['rollback']:
                layer_candidates[key]['s_score'] = a * (item['before_train_score'] + 1) + b * (
                        item['before_val_score'] + 1)
            else:
                layer_candidates[key]['s_score'] = a * (item['train_score'] + 1) + b * (
                        item['val_score'] + 1)
                # layer_candidates[key]['s_score'] = a * (item['prune_train_score'] + 1) + b * (
                #         item['prune_val_score'] + 1)
            layer_candidates[key]['resource_score'] = max(1, (item['encoder_macs'] + item['macs']) / floats_scale)
            layer_candidates[key]['score'] = item['s_score'] / layer_candidates[key]['resource_score']
            encoder_macs, encoder_params = clever_format([item['encoder_macs'], item['encoder_params']], "%.2f")
            macs, params = clever_format([item['macs'], item['params']], "%.2f")
            all_macs, all_params = clever_format([(item['macs'] + item['encoder_macs']),
                                                  (item['params'] + item['encoder_params'])],
                                                 "%.2f")
            if item['pre_macs'] is not None:
                macs_pre, params_pre = clever_format([item['pre_macs'], item['pre_params']], "%.2f")
                layer_candidates[key]['pre_macs_p'] = macs_pre
                layer_candidates[key]['pre_params_p'] = params_pre
            else:
                layer_candidates[key]['pre_macs_p'] = 0
                layer_candidates[key]['pre_params_p'] = 0
            layer_candidates[key]['encoder_macs_p'] = encoder_macs
            layer_candidates[key]['encoder_params_p'] = encoder_params
            layer_candidates[key]['macs_p'] = macs
            layer_candidates[key]['params_p'] = params
            layer_candidates[key]['all_macs_p'] = all_macs
            layer_candidates[key]['all_params_p'] = all_params
            layer_candidates[key]['grad_score'] = item['prune_train_score'] - item['prune_test_score']
            layer_candidates[key]['train_rate'] = (item['train_score'] - item['before_train_score']) / (
                    item['anchor_score'] - item['before_train_score'])

            layer_candidates[key]['test_rate'] = (item['test_score'] - item['before_test_score']) / (
                    item['anchor_score'] - item['before_test_score'])

        thres_unit = 0.00
        candidates = {}

        for key, item in layer_candidates.items():
            con = True
            # if not item['rollback']:
            #     if not item['prune_train_score'] >= self.last_train_score:
            #         con = False
            #     if not item['prune_train_score'] >= (item['before_train_score']):
            #         con = False
            #     if self.cfg.RSR.BEST_VALIDATION:
            #         if not item['prune_val_score'] >= (self.last_val_score - thres_unit):
            #             con = False
            #         if not item['prune_val_score'] >= (item['before_val_score'] - thres_unit):
            #             con = False
            if not item['rollback']:
                if not item['train_score'] >= self.last_train_score:
                    con = False
                if not item['train_score'] >= (item['before_train_score']):
                    con = False
                if self.cfg.RSR.BEST_VALIDATION:
                    if not item['val_score'] >= (self.last_val_score - thres_unit):
                        con = False
                    if not item['val_score'] >= (item['before_val_score'] - thres_unit):
                        con = False

            if self.backbone_layer == 1:
                con = True

            # SSR_only
            if True:
                candidates[key] = item

            # if con is True:
            #     candidates[key] = item

        print('After filter selection, layer candidates: {}'.format([c for c in candidates.keys()]))
        Notes.write('After filter selection, layer candidates: {}'.format([c for c in candidates.keys()]))

        if len(candidates) == 0:
            return None

        if self.cfg.RSR.MAX_RESOURCE is not None:
            for key in list(candidates.keys()):
                item = candidates[key]
                if (item['macs'] + item['encoder_macs']) + self.total_macs > self.cfg.RSR.MAX_RESOURCE:
                    candidates.pop(key)
            if len(candidates) == 0:
                print('After resource check, no layer candidate meet requirements')
                Notes.write('After resource check, no layer candidate meet requirements')
                self.break_switch = False
                return None
            print('After resource check, layer candidates: {}'.format([c for c in candidates.keys()]))
            Notes.write('After resource check, layer candidates: {}'.format([c for c in candidates.keys()]))

        if self.cfg.RSR.RESOURCE_SWITCH:
            bestid = [k for k, v in sorted(candidates.items(),
                                           key=lambda item: item[1]['score'], reverse=True)][0]
        else:
            bestid = [k for k, v in sorted(candidates.items(),
                                           key=lambda item: item[1]['s_score'], reverse=True)][0]
        return bestid

    def select_layer(self, dictL, input_og, label, test_x, test_y, val_x=None, val_y=None, quick=False, **kwargs):
        # if '3' in kwargs['nid']:
        #     print('')
        layer_id = kwargs['lid']

        # layer  dictionary
        layer_dict = OrderedDict()

        # copy input
        input = copy.deepcopy(input_og.detach().cpu())
        test_x_og = copy.deepcopy(test_x.detach().cpu())

        # get anchor and anchor's label
        anchor = dictL['anchor']
        anchor_label = dictL['anchor_label']
        anchor_inter = dictL['anchor_inter']

        # add resizer
        input_size = tuple(input.size()[2:]) if len(input.size()) > 3 else tuple(input.size()[1:])
        first_layer = True if layer_id == 0 and input_size[0] != input_size[1] else False
        if layer_id != 0 and input_size != dictL['input_size'][1:] and self.cfg.RSR.RESIZER_SWITCH:
            # if layer_id != 0:
            # add
            resize_transformer = ResizeTrans(input_size, dictL['input_size'], is_1d=self.cfg.RSR.Conv1D)
            # resize_transformer = ResizeTrans(input_size, [int(dictL['input_size'][-1] * 0.5)])
            resizer = resize_transformer.get_resizer()
            layer_dict['resizer'] = resizer
            input = resizer(input)
            test_x = resizer(test_x)
            val_x = resizer(val_x)
        else:
            layer_dict['resizer'] = None

        # quick check
        print('Check {} >>> '.format(kwargs['nid']))

        Notes.write('Check {} >>> '.format(kwargs['nid']))
        print('Train S-score: ', end='')
        before_train_score, og_model = quick_check(input, label, anchor, anchor_label,
                                                   next_input_size=dictL['input_size'],
                                                   head=dictL['head'], n_comp=self.n_comp, norm_mode=self.norm_mode,
                                                   first_layer=first_layer, regression=self.regression,
                                                   backbone_input=self.cfg.DATALOADER.RESIZE,
                                                   dic_key='before', layer_id=layer_id)
        torch.cuda.empty_cache()
        print('Test S-score: ', end='')
        before_test_score, _ = quick_check(test_x, test_y, anchor, anchor_label, next_input_size=dictL['input_size'],
                                           head=dictL['head'], n_comp=self.n_comp, norm_mode=self.norm_mode,
                                           model=og_model, regression=self.regression)
        torch.cuda.empty_cache()
        print('Val S-score: ', end='')
        before_val_score, _ = quick_check(val_x, val_y, anchor, anchor_label, next_input_size=dictL['input_size'],
                                          head=dictL['head'], n_comp=self.n_comp, norm_mode=self.norm_mode,
                                          model=og_model, regression=self.regression)
        torch.cuda.empty_cache()
        layer_dict['before_train_score'] = before_train_score
        layer_dict['before_test_score'] = before_test_score
        layer_dict['before_val_score'] = before_val_score

        if quick:
            return layer_dict

        if 'estimated_score' in dictL.keys():
            self.estimate_score_history[self.backbone_layer] = round(dictL['estimated_score'], 4)

        # initialize bn running mean and vars for sensing
        bn_init(x=input, next_input_size=dictL['input_size'], head=dictL['head'], model=og_model)
        og_head = copy.deepcopy(dictL['head'])
        og_head.eval()

        print('After BN Train S-score: ', end='')
        bn_train_score, _ = quick_check(input, label, anchor, anchor_label,
                                        next_input_size=dictL['input_size'], model=og_model,
                                        head=dictL['head'], n_comp=self.n_comp, norm_mode=self.norm_mode,
                                        regression=self.regression)
        print('Test S-score: ', end='')
        bn_test_score, _ = quick_check(test_x, test_y, anchor, anchor_label, next_input_size=dictL['input_size'],
                                       head=dictL['head'], n_comp=self.n_comp, norm_mode=self.norm_mode,
                                       model=og_model, regression=self.regression)
        print('Val S-score: ', end='')
        bn_val_score, _ = quick_check(val_x, val_y, anchor, anchor_label, next_input_size=dictL['input_size'],
                                      head=dictL['head'], n_comp=self.n_comp, norm_mode=self.norm_mode,
                                      model=og_model, regression=self.regression)
        layer_dict['bn_train_score'] = bn_train_score
        layer_dict['bn_test_score'] = bn_test_score
        layer_dict['bn_val_score'] = bn_val_score

        layer_dict['prehead_og'] = og_model
        layer_dict['head_og'] = og_head

        # add prehead
        if self.fit_schema['prehead']:
            trans_mode = self.fit_schema['prehead']
            prehead_trans = self.build_transformer(trans_mode, input, label, anchor, anchor_label, head=dictL['head'],
                                                   input_size=dictL['input_size'], layer_id=layer_id,
                                                   test_x=test_x, n_comp=self.n_comp, test_y=test_y,
                                                   first_layer=first_layer,
                                                   backbone_input=self.cfg.DATALOADER.RESIZE)
            prehead = prehead_trans.get_prehead()
            macs, params = prehead_trans.get_macs_params()
            macs_pre, params_pre = prehead_trans.get_mac_params_pre()
            layer_dict['pre_macs'] = macs_pre
            layer_dict['pre_params'] = params_pre
            layer_dict['anchor_score'] = prehead_trans.get_anchor_score()
            layer_dict['encoder_macs'] = macs
            layer_dict['encoder_params'] = params
            layer_dict['prehead'] = prehead.eval()
            layer_dict['pca_fun'] = prehead_trans.pca
            layer_dict['time'] = prehead_trans.get_time_dict()
            layer_dict['epoch_loss'] = prehead_trans.get_epoch_loss()

        else:
            layer_dict['prehead'] = None

        # add head layer
        dictL['head'] = dictL['head'].cpu()
        dictL['head'].eval()
        layer_dict['head'] = dictL['head']

        print('After repairing S-score:')
        print('Train: ', end='')
        layer_dict['train_score'], _ = quick_check(input, label, anchor, anchor_label, head=layer_dict['head'],
                                                   model=layer_dict['prehead'], norm_mode=self.norm_mode,
                                                   regression=self.regression, dic_key='after', layer_id=layer_id)
        print('Test: ', end='')
        layer_dict['test_score'], _ = quick_check(test_x, test_y, anchor, anchor_label, head=layer_dict['head'],
                                                  model=layer_dict['prehead'], norm_mode=self.norm_mode,
                                                  regression=self.regression)
        print('Val: ', end='')
        layer_dict['val_score'], _ = quick_check(val_x, val_y, anchor, anchor_label, head=layer_dict['head'],
                                                 model=layer_dict['prehead'], norm_mode=self.norm_mode,
                                                 regression=self.regression)

        if self.cfg.RSR.BEST_VALIDATION:
            a = 0.5
            b = 0.5
        else:
            a = 1
            b = 0

        layer_dict['before_score'] = a * (layer_dict['before_train_score'] + 1) + b * (
                layer_dict['before_val_score'] + 1)
        layer_dict['after_score'] = a * (layer_dict['train_score'] + 1) + b * (layer_dict['val_score'] + 1)

        if layer_dict['after_score'] < layer_dict['before_score'] and self.cfg.RSR.ROLLBACK_SWITCH:
            print('@@@ Repaired test score is worse than og test score, rollback to init model @@@')
            Notes.write('@@@ Repaired test score is worse than og test score, rollback to init model @@@')
            layer_dict['prehead'] = og_model
            layer_dict['head'] = og_head
            layer_dict['rollback'] = True
        else:
            layer_dict['rollback'] = False

        # Prototype to check acc before pruning
        train_acc_og, test_acc_og = prototype_test(input_og, label, test_x_og, test_y, head=layer_dict['head'],
                                                   model=layer_dict['prehead'],
                                                   resizer=layer_dict['resizer'])
        layer_dict['train_acc'], layer_dict['test_acc'] = train_acc_og, test_acc_og
        print('>>> Prototype accuracy: train={:.4f}, test={:.4f}'.format(train_acc_og, test_acc_og))

        # pruning
        if self.fit_schema['pruner']:
            prune_input = layer_dict['prehead'](input)
            trans_mode = self.fit_schema['pruner']
            stime = time.time()
            pruner = self.build_transformer(trans_mode=trans_mode, input=prune_input, label=label,
                                            anchor_input=anchor, anchor_label=anchor_label, anchor_inter=anchor_inter,
                                            head=layer_dict['head'], n_comp=self.n_comp)
            pruner.prune()
            prune_time = time.time() - stime
            layer_dict['time']['prune'] = prune_time
            macs, params = pruner.get_macs_params()
            layer_dict['pruned_head'] = pruner.get_pruned_head().eval()
            layer_dict['macs'] = macs
            layer_dict['params'] = params
            layer_dict['prune_masks'] = pruner.get_masks()
        else:
            layer_dict['pruned_head'] = dictL['head']
            layer_dict['macs'] = 0
            layer_dict['params'] = 0
            layer_dict['prune_masks'] = None

        print('After pruning S-score:')
        print('Train: ', end='')
        layer_dict['prune_train_score'], _ = quick_check(input, label, anchor, anchor_label,
                                                         head=layer_dict['pruned_head'],
                                                         model=layer_dict['prehead'], norm_mode=self.norm_mode,
                                                         prune_mask=layer_dict['prune_masks'],
                                                         regression=self.regression, dic_key='pruned',
                                                         layer_id=layer_id)
        print('Test: ', end='')
        layer_dict['prune_test_score'], _ = quick_check(test_x, test_y, anchor, anchor_label,
                                                        head=layer_dict['pruned_head'],
                                                        model=layer_dict['prehead'], norm_mode=self.norm_mode,
                                                        prune_mask=layer_dict['prune_masks'],
                                                        regression=self.regression)
        print('Val: ', end='')
        layer_dict['prune_val_score'], _ = quick_check(val_x, val_y, anchor, anchor_label,
                                                       head=layer_dict['pruned_head'],
                                                       model=layer_dict['prehead'], norm_mode=self.norm_mode,
                                                       prune_mask=layer_dict['prune_masks'],
                                                       regression=self.regression)

        # Prototype to check acc before pruning
        train_acc_prune, test_acc_prune = prototype_test(input_og, label, test_x_og, test_y,
                                                         head=layer_dict['pruned_head'],
                                                         model=layer_dict['prehead'], resizer=layer_dict['resizer'])
        layer_dict['prune_train_acc'], layer_dict['prune_test_acc'] = train_acc_prune, test_acc_prune
        print('>>> Prototype accuracy after prune: train={:.4f}, test={:.4f}'.format(train_acc_prune, test_acc_prune))

        # add tail layer
        if dictL['tail']:
            layer_dict['tail'] = dictL['tail']

        return layer_dict

    def check_score(self, layer_pool, train_x, train_y, test_x, test_y, val_x=None, val_y=None, **kwargs):
        min_percentage = 0.2
        max_percentage = 0.5
        x = self.backbone_layer
        # 1. fit curve
        if len(self.score_history) >= 2 and (x - 1) in self.score_history.keys():
            pred = self.estimate_score_history[x - 1]
            true = self.score_history[x - 1]
            # a = np.array([self.score_history[x - 1], self.score_history[x - 2]])
            # b = np.array([self.estimate_score_history[x - 1], self.score_history[x - 2]])
            # c = np.array([0.00, self.score_history[x - 2]])
            # cossim = 1 - np.dot(a, b) / (norm(a) * norm(b))
            # a = self.score_history[x - 1] - self.score_history[x - 2]
            # b = self.estimate_score_history[x - 1] - self.score_history[x - 2]
            # self.pick_range = self.pick_range + ((pred - true) / pred) * self.pick_range
            if pred < true:
                # base_cossim = 1 - np.dot(a, c) / (norm(a) * norm(c))
                # self.pick_range = self.pick_range - (1-(b/a)) * self.pick_range
                # self.pick_range = self.pick_range - (cossim / base_cossim) * self.pick_range
                rate = min((abs(true - pred) / pred), 1)
                self.pick_range = self.pick_range - rate * self.pick_range
            else:
                rate = min((abs(true - pred) / pred), 1)
                # base_cossim = 1 - np.dot(b, c) / (norm(b) * norm(c))
                # self.pick_range = self.pick_range + (1-(a/b)) * self.pick_range
                # self.pick_range = self.pick_range + (cossim / base_cossim) * self.pick_range
                self.pick_range = self.pick_range + rate * self.pick_range

            print('Truth score history: {}'.format(self.score_history))
            print('Estimate score history: {}, changed percentage={:.2f}%'.format(self.estimate_score_history,
                                                                                  rate * 100))

            Notes.write('Truth score history: {}'.format(self.score_history))
            Notes.write('Estimate score history: {}, changed percentage={:.4f}'.format(self.estimate_score_history,
                                                                                       (abs(true - pred) / pred)))

        self.pick_range = min(max(min_percentage, self.pick_range), max_percentage)
        top_n = int(max((len(layer_pool) * self.pick_range) // 1 + 1, 2))
        print('Pick range: {:.4f}, num of candidates: {}'.format(self.pick_range, top_n))

        Notes.write('Pick range: {:.4f}, num of candidates: {}'.format(self.pick_range, top_n))
        popt = fit_curve(self.rate_data)
        layers = defaultdict()
        rate = func(x, *popt)
        # self.estimate_score_history[self.backbone_layer] = round(rate, 4)
        for nid, dictL in layer_pool.items():
            lid = int(nid.split('_')[1])
            layer_dict = self.select_layer(dictL, train_x, train_y, test_x=test_x,
                                           test_y=test_y, val_x=val_x, val_y=val_y, quick=True, lid=lid, nid=nid)
            anchor_score = calculate_anchor_score(dictL, norm_mode=self.norm_mode, num_classes=len(np.unique(train_y)),
                                                  n_comp=self.n_comp)
            layer_dict['anchor_score'] = anchor_score
            layer_dict['estimated_rate'] = rate
            layer_dict['estimated_score'] = layer_dict['before_train_score'] + rate * (
                    layer_dict['anchor_score'] - layer_dict['before_train_score'])

            layer_pool[nid]['estimated_rate'] = layer_dict['estimated_rate']
            layer_pool[nid]['estimated_score'] = layer_dict['estimated_score']

            layers[nid] = layer_dict

        scores = {k: {'before_train_score': round(v['before_train_score'], 4),
                      'anchor_score': round(v['anchor_score'], 4),
                      'estimated_rate': round(v['estimated_rate'], 3),
                      'estimated_score': round(v['estimated_score'], 4),
                      } for k, v in layers.items()}
        scores = json.dumps(scores, indent=4)
        print(scores)
        Notes.write('Quick Check:')
        Notes.write(scores)

        top_layer_keys = [k for k, v in sorted(layers.items(),
                                               key=lambda item: item[1]['estimated_score'], reverse=True)][:top_n]
        print('Selected top layers: {}'.format(top_layer_keys))
        Notes.write('Selected top layers: {}'.format(top_layer_keys))
        select_layer_pool = {key: layer_pool[key] for key in top_layer_keys}

        return select_layer_pool, layers

    @torch.no_grad()
    def check_metric(self):
        for train_x, train_y in self.trainloader:
            train_x = self.infer_backbone(train_x)
        train_x = train_x.cpu().numpy()
        train_y = train_y.cpu().numpy()

        for test_x, test_y in self.testloader:
            test_x = self.infer_backbone(test_x)

        test_x = test_x.cpu().numpy()
        test_y = test_y.cpu().numpy()

        train_prototypes = class_centroids(train_x, train_y)
        test_prototypes = class_centroids(test_x, test_y)
        distance = np.asarray(paired_distances(train_prototypes, test_prototypes))
        print('Class centroid distance between train and test: {}'.format(distance))
        # Notes.write('Class centroid distance between train and test: {}'.format(distance))

        train_inter = inter_distance(train_prototypes, np.arange(0, len(train_prototypes)))
        test_inter = inter_distance(test_prototypes, np.arange(0, len(test_prototypes)))
        print('Train inter distance for each class: {}, Mean: {:.4f}'.format(train_inter, np.mean(train_inter)))
        print('Test inter distance for each class: {}, Mean: {:.4f}'.format(test_inter, np.mean(test_inter)))
        # Notes.write('Train inter distance for each class: {}, Mean: {:.4f}'.format(train_inter, np.mean(train_inter)))
        # Notes.write('Test inter distance for each class: {}, Mean: {:.4f}'.format(test_inter, np.mean(test_inter)))

        train_score = class_silhouette_score(train_x, train_y, 4, regression=self.regression)
        train_mean_score = np.round(np.mean(list(train_score.values())), 4)
        test_score = class_silhouette_score(test_x, test_y, 4, regression=self.regression)
        test_mean_score = np.round(np.mean(list(test_score.values())), 4)
        print('Train S-score for each class: {} , Mean: {:.4f}'.format(train_score, train_mean_score))
        print('Test S-score for each class: {} , Mean: {:.4f}'.format(test_score, test_mean_score))
        # Notes.write('Train S-score for each class: {} , Mean: {:.4f}'.format(train_score, train_mean_score))
        # Notes.write('Test S-score for each class: {} , Mean: {:.4f}'.format(test_score, test_mean_score))

    # @torch.no_grad()
    def scorer(self, method='KNN', **kwargs):
        if 'Linear' in method:
            train_acc, test_acc = self.scorer_classifier()
            return train_acc, test_acc

        pca = copy.deepcopy(self.pca)
        if 'prune_mask' in kwargs:
            prune_mask = kwargs['prune_mask']
            if prune_mask is not None:
                pca.components_ = pca.components_[:, prune_mask]
                pca.mean_ = pca.mean_[prune_mask]
                pca.n_features_in_ = len(prune_mask)

        # self.backbone.eval()

        for train_x, train_y in self.trainloader:
            output = self.infer_backbone(train_x)
        X = output.detach().cpu().numpy()
        y = train_y.detach().cpu().numpy()
        if 'pca' in method:
            X = pca.transform(X)

        if 'KNN' in method:
            # classifier = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
            classifier = KNeighborsClassifier(n_neighbors=5)
        elif 'SVM_linear' in method:
            classifier = SVC(kernel='linear')
            # classifier = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        elif 'SVM_rbf' in method:
            classifier = SVC(kernel='rbf')
            # classifier = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
        else:
            raise NotImplementedError("Not Implemented Error")

        classifier.fit(X, y)
        predict_train = classifier.predict(X)
        train_acc = accuracy_score(y, predict_train)
        train_class_acc = class_accuracy(y, predict_train)

        for test_x, test_y in self.testloader:
            test_x = self.infer_backbone(test_x)

        test_X = test_x.detach().cpu().numpy()
        test_y = test_y.detach().cpu().numpy()
        if 'pca' in method:
            test_X = pca.transform(test_X)

        predict_test = classifier.predict(test_X)
        test_acc = accuracy_score(test_y, predict_test)
        test_class_acc = class_accuracy(test_y, predict_test)

        print('=====================================')
        print('Train accuracy is {:.4f}, Test accuracy is {:.4f}'.format(train_acc, test_acc))
        print('Train accuracy each class is {}'.format(train_class_acc))
        print('Test accuracy each class is {}'.format(test_class_acc))
        print('=====================================')
        if 'pca' in method:
            Notes.write('PCA: Train accuracy is {:.4f}, Test accuracy is {:.4f}'.format(train_acc, test_acc))
        else:
            Notes.write('Train accuracy is {:.4f}, Test accuracy is {:.4f}'.format(train_acc, test_acc))
        return train_acc, test_acc

    def scorer_regression(self, method='SVR', og=False, **kwargs):

        pca = copy.deepcopy(self.pca)
        if 'prune_mask' in kwargs:
            prune_mask = kwargs['prune_mask']
            if prune_mask is not None:
                pca.components_ = pca.components_[:, prune_mask]
                pca.mean_ = pca.mean_[prune_mask]
                pca.n_features_in_ = len(prune_mask)

        for train_x, train_y in self.trainloader:
            output = self.infer_backbone(train_x, og=og)
        if method != 'linear':
            X = output.detach().cpu().numpy()
            y = train_y.detach().cpu().numpy()
        else:
            X = output
            y = train_y

        if 'pca' in method:
            X = pca.transform(X)

        for test_x, test_y in self.testloader:
            test_x = self.infer_backbone(test_x, og=og)
        if method != 'linear':
            test_X = test_x.detach().cpu().numpy()
            test_y = test_y.detach().cpu().numpy()
        else:
            test_X = test_x
            test_y = test_y
        if 'pca' in method:
            test_X = pca.transform(test_X)

        if 'KNN' in method:
            classifier = KNeighborsRegressor(n_neighbors=2)
            classifier.fit(X, y)
            predict_train = classifier.predict(X)

            predict_test = classifier.predict(test_X)
        elif 'SVR' in method:
            from sklearn.svm import SVR
            classifier = SVR()
            classifier.fit(X, y)
            predict_train = classifier.predict(X)
            predict_test = classifier.predict(test_X)

        elif 'linear' in method:
            linear = kwargs['linear']
            predict_train = linear(X)
            predict_test = linear(test_X)

        else:
            raise NotImplementedError("Not Implemented Error")

        diff_train = predict_train - y
        me_train = diff_train.mean()
        std_train = diff_train.std()
        train_score = (me_train, std_train)

        diff_test = predict_test - test_y
        me_test = diff_test.mean()
        std_test = diff_test.std()
        test_score = (me_test, std_test)

        print('=====================================')
        print('Train {:.4f} +- {:.4f}, Test is {:.4f}+-{:.4f}'.format(me_train, std_train, me_test, std_test))
        print('=====================================')
        if 'pca' in method:
            Notes.write('PCA Train {:.4f} +- {:.4f}, Test is {:.4f}+-{:.4f}'.format(me_train, std_train,
                                                                                    me_test, std_test))
        else:
            Notes.write('Train {:.4f} +- {:.4f}, Test is {:.4f}+-{:.4f}'.format(me_train, std_train, me_test, std_test))
        return train_score, test_score

    def scorer_classifier(self, trained=False, linear=None, og=False):
        self.backbone.eval()

        # prepare data
        for train_x, train_y in self.trainloader:
            output = self.infer_backbone(train_x, og=og)
        X = output.detach().cpu()
        y = train_y.detach().cpu()
        num_classes = len(np.unique(y))

        for test_x, test_y in self.testloader:
            test_x = self.infer_backbone(test_x, og=og)
        test_X = test_x.detach().cpu()
        test_y = test_y.detach().cpu()

        if not trained:
            model = nn.Linear(in_features=X.shape[1], out_features=num_classes)
            trainer = SimpleTrainer(model, X, y)
            trainer.fit()
            predict_train = trainer.predict(X)
            predict_test = trainer.predict(test_X)
        else:
            if linear is not None:
                predict_train = linear(X)
                predict_test = linear(test_X)
                predict_train = predict_train.argmax(dim=1, keepdim=True)
                predict_test = predict_test.argmax(dim=1, keepdim=True)
            else:
                predict_train = X.argmax(dim=1, keepdim=True)
                predict_test = test_X.argmax(dim=1, keepdim=True)

        train_acc = accuracy_score(y, predict_train)
        train_class_acc = class_accuracy(y, predict_train)

        test_acc = accuracy_score(test_y, predict_test)
        test_class_acc = class_accuracy(test_y, predict_test)

        Notes.write('Train accuracy is {:.4f}, Test accuracy is {:.4f}'.format(train_acc, test_acc))

        print('=====================================')
        print('Train accuracy: {:.2f}%, Test accuracy: {:.2f}%'.format(train_acc * 100, test_acc * 100))
        print('Train accuracy each class is {}'.format(train_class_acc))
        print('Test accuracy each class is {}'.format(test_class_acc))
        print('=====================================')
        return train_acc, test_acc

    def infer_backbone(self, input, flatten=False, og=False):
        output = self.backbone(input)
        if flatten is True:
            output = output.view(output.size(0), -1)
        else:
            if not og:
                output = mmc(output)
                output = F.normalize(output)
        return output

    def build_transformer(self, trans_mode, input, label, anchor_input=None, anchor_label=None, **kwargs):
        if trans_mode == 'Pruner':
            anchor_inter = kwargs['anchor_inter']
            head = kwargs['head']
            tran = PruneTrans(input, label, anchor_input, anchor_label, anchor_inter, head, n_comp=self.n_comp,
                              norm_mode=self.norm_mode, max_range=self.cfg.RSR.MAX_PRUNE_RANGE,
                              regression=self.regression)

        elif trans_mode == 'Repair':
            input_size = kwargs['input_size']
            layer_id = kwargs['layer_id']
            test_x = kwargs['test_x']
            test_y = kwargs['test_y']
            first_layer = kwargs['first_layer']
            backbone_input = kwargs['backbone_input']
            num_episode = self.cfg.TRAN.NUM_EPISODE
            num_episode_rot = self.cfg.TRAN.ROT_NUM_EPISODE
            mode = self.cfg.TRAN.ANCHOR_SELECT_MODE
            rm = self.cfg.TRAN.RM
            best_anchor_mode = self.cfg.RSR.BEST_ANCHOR

            if 'mmc' in self.cfg.RSR.LOSS_MODE:
                tran_class = MMCTrans
            else:
                tran_class = RepairTrans

            tran = tran_class(input, label, anchor_input, anchor_label, kwargs['head'], out_size=input_size,
                              mode=mode, n_comp=self.n_comp, test_x=test_x, test_y=test_y, rm=rm,
                              layer_id=layer_id, norm_mode=self.norm_mode, num_episode=num_episode,
                              best_anchor=best_anchor_mode, first_layer=first_layer, backbone_input=backbone_input,
                              num_episode_rot=num_episode_rot, regression=self.cfg.RSR.REGRESSION,
                              loss_mode=self.cfg.RSR.LOSS_MODE)

        elif trans_mode == 'Norm':
            input_size = kwargs['input_size']
            test_x = kwargs['test_x']
            test_y = kwargs['test_y']
            num_episode = self.cfg.TRAN.NUM_EPISODE
            tran = NormTrans(input, label, anchor_input, anchor_label, test_x=test_x, test_y=test_y,
                             head=kwargs['head'], num_episode=num_episode, norm_mode=self.norm_mode,
                             out_size=input_size)
        elif trans_mode == 'cnn':
            tran = LastTrans(x=input, y=label, num_episode=self.cfg.TRAN.NUM_EPISODE, norm_mode=self.norm_mode)
        else:
            raise NotImplementedError
        return tran
