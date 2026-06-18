from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from xtransfer.pytorch_trainer import Module
from xtransfer.evaluation import Classification
from utils.meters import AverageMeter, accuracy
from utils import qmul, set_random_seed


class EarlyStopping(object):
    def __init__(self, mode='min', min_break_epoch=50, min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.min_break_epoch = min_break_epoch
        self.num_epochs = 0
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        self.num_epochs += 1
        if self.num_epochs <= self.min_break_epoch:
            return False

        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class OXiodLinear(nn.Module):
    def __init__(self, in_dim=256):
        super(OXiodLinear, self).__init__()
        set_random_seed(5)
        self.linear1 = nn.Linear(in_dim, 3)
        self.linear2 = nn.Linear(in_dim, 4)

    def forward(self, x):
        y1 = self.linear1(x)
        y2 = self.linear2(x)

        return y1, y2


class SimpleTrainer:
    def __init__(self, model, X, y, num_epochs=200, loss_func=None):
        self.model = model
        self.X = X
        self.y = y
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.95)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.8)
        if loss_func is None:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = loss_func
        print("Model Structure:")
        print(self.model)

    def fit(self):
        for t in range(self.num_epochs):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(self.X)

            # loss
            loss = self.loss_func(y_pred, self.y)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()

            # update step
            self.optimizer.step()
            self.scheduler.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct = pred.eq(self.y.view_as(pred)).sum().item()
            acc = 100. * correct / len(self.y)
            # print
            if (t + 1) % 10 == 0:
                print(
                    'Epoch {:05} >>> Training loss: {:.5f}, Training accuracy: {:.2f}%'.format(t + 1, loss.item(), acc))

    def predict(self, x):
        y_pred = self.model(x)
        pred = y_pred.argmax(dim=1, keepdim=True)
        return pred


class CustomMultiLossLayer(nn.Module):
    def __init__(self):
        super(CustomMultiLossLayer, self).__init__()
        self.register_buffer('log_vars', torch.zeros(2, requires_grad=True))

    def forward(self, ys_true, ys_pred):
        loss = 0
        precision = torch.exp(-self.log_vars[0])
        loss += precision * torch.mean(torch.abs(ys_true[0] - ys_pred[0]), dim=-1) + self.log_vars[0]
        precision = torch.exp(-self.log_vars[1])
        loss += precision * self.quaternion_mean_multiplicative_error(ys_true[1], ys_pred[1]) + self.log_vars[1]
        return torch.mean(loss)

    def quaternion_mean_multiplicative_error(self, y_true, y_pred):
        q = y_pred / torch.sqrt(torch.sum(torch.square(y_pred), dim=-1, keepdim=True))
        q_prod = qmul(q, torch.multiply(y_true, torch.Tensor([1.0, -1.0, -1.0, -1.0]).cuda()))

        w, xyz = torch.split(q_prod, split_size_or_sections=[1, 3], dim=-1)
        q_res = torch.abs(torch.multiply(torch.Tensor([2.0]).cuda(), xyz))
        return torch.mean(q_res)


class OXiodTrainer:
    def __init__(self, model, X, y, num_epochs=100):
        self.model = model
        self.X = X
        self.y = y
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_func = CustomMultiLossLayer()
        print("Model Structure:")
        print(self.model)

    def fit(self):
        for t in range(self.num_epochs):
            # Forward pass: Compute predicted y by passing x to the model
            ypred1, ypred2 = self.model(self.X)
            ytrue1 = self.y[:, :3]
            ytrue2 = self.y[:, 3:]

            # loss
            loss = self.loss_func([ytrue1, ytrue2], [ypred1, ypred2])

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()

            # update step
            self.optimizer.step()

            # print
            if (t + 1) % 10 == 0:
                print('Epoch {:05} >>> Training loss: {:.5f}'.format(t + 1, loss.item()))

    def predict(self, x):
        ypred_1, ypred_2 = self.model(x)
        return ypred_1, ypred_2


class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.labels = label
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Classifier(Module):
    def __init__(self, fdim_in, num_classes, train_x, train_y, test_x, test_y):
        super().__init__()
        self.l1 = torch.nn.Linear(fdim_in, num_classes)
        self.evaluator = Classification()
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.top1 = AverageMeter()
        self.best_acc = 0

    def forward(self, x):
        return self.l1(x)

    def training_step(self, batch, batch_num):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        batch_acc, _ = accuracy(y_hat, y, topk=(1, 5))
        self.top1.update(batch_acc.item(), 35)
        return {'loss': loss, 'accuracy': batch_acc}

    def validation_step(self, batch, batch_num):
        x, y = batch
        output = self.forward(x)
        self.evaluator.process(output, y)
        return {'val_loss': F.cross_entropy(output, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        res = self.evaluator.evaluate()
        if res['accuracy'] > self.best_acc:
            self.best_acc = res['accuracy']
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def on_epoch_end(self, epoch):
        print("train accuracy={}".format(self.top1.avg))

    def on_epoch_start(self, epoch):
        self.top1.reset()

    def train_dataloader(self):
        return DataLoader(CustomDataset(data=self.train_x, label=self.train_y), batch_size=35)

    def val_dataloader(self):
        return DataLoader(CustomDataset(data=self.test_x, label=self.test_y), batch_size=350)


class Net(Module):
    def __init__(self, backbone, fim_in, num_classes, train_loader=None, test_loader=None):
        super(Net, self).__init__()
        self.backbone = backbone
        self.evaluator = Classification()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.head = torch.nn.Linear(fim_in, num_classes)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.top1 = AverageMeter()
        self.best_acc = 0

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.head(x)
        return out

    def freeze_backbone(self):
        for name, module in self.backbone.named_children():
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def training_step(self, batch, batch_num):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        batch_acc, _ = accuracy(y_hat, y, topk=(1, 5))
        self.top1.update(batch_acc.item(), 35)
        return {'loss': loss, 'accuracy': batch_acc}

    def validation_step(self, batch, batch_num):
        x, y = batch
        output = self.forward(x)
        self.evaluator.process(output, y)
        return {'val_loss': F.cross_entropy(output, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        res = self.evaluator.evaluate()
        if res['accuracy'] > self.best_acc:
            self.best_acc = res['accuracy']
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.02)
        return torch.optim.Adam(self.head.parameters(), lr=0.01)

    def on_epoch_end(self, epoch):
        print("train accuracy: {:.4f}".format(self.top1.avg))

    def on_epoch_start(self, epoch):
        self.top1.reset()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader


class NetLoader(Module):
    def __init__(self, model, train_loader=None, test_loader=None, **kwargs):
        super(NetLoader, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.top1 = AverageMeter()
        self.best_acc = 0
        self.evaluator = Classification()
        self.header = None
        if 'header' in kwargs.keys():
            self.header = kwargs['header']

    def forward(self, x):
        if self.header:
            x = self.header(x)
        out = self.model(x)
        return out

    def freeze_backbone(self):
        for name, module in self.model.backbone.named_children():
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def release_backbone(self):
        for name, module in self.model.backbone.named_children():
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        for name, module in self.model.classifier.named_children():
            module.train()
            for p in module.parameters():
                p.requires_grad = True

    def training_step(self, batch, batch_num):
        # self.freeze_backbone()
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        batch_acc, _ = accuracy(y_hat, y, topk=(1, 5))
        self.top1.update(batch_acc.item(), 35)
        return {'loss': loss, 'accuracy': batch_acc}

    def validation_step(self, batch, batch_num):
        x, y = batch
        output = self.forward(x)
        self.evaluator.process(output, y)
        return {'val_loss': F.cross_entropy(output, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        res = self.evaluator.evaluate()
        if res['accuracy'] > self.best_acc:
            self.best_acc = res['accuracy']
        print('Test accuracy: {:.4f}'.format(res['accuracy']))
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.02)
        # return torch.optim.Adam(self.model.parameters(), lr=0.01)
        return torch.optim.SGD(self.model.classifier.parameters(), lr=0.001, momentum=0.95)
        # return torch.optim.SGD(self.model.parameters(), lr=0.01,
        #                         momentum=0.9, dampening=0.9, weight_decay=0.001)
        # return torch.optim.SGD(self.model.classifier.parameters(), lr=0.01,
        #                         momentum=0.9, dampening=0.9, weight_decay=0.001)

    def on_epoch_end(self, epoch):
        print("Train accuracy: {:.4f}".format(self.top1.avg))

    def on_epoch_start(self, epoch):
        self.top1.reset()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

