import torch, random, itertools as it, numpy as np, random
import torch.nn.functional as F
from numpy.linalg import norm
from torch import nn
from functools import partial
from torch.autograd import Variable


class TupleSampler():
    """
    Container for all sampling methods that can be used in conjunction with the respective loss functions.
    Based on batch-wise sampling, i.e. given a batch of training data, sample useful data tuples that are
    used to train the network more efficiently.
    """

    def __init__(self, method='random', **kwargs):
        """
        Args:
            method: str, name of sampling method to use.
        Returns:
            Nothing!
        """
        self.method = method
        if method == 'anchor_npair':
            self.give = self.anchor_npair
        elif method == 'anchor_neg_cosine':
            self.give = self.anchor_neg_cosine
        elif method == 'anchor_npair_regression':
            self.give = self.anchor_npair_regression

    def anchor_neg_cosine(self, batch, labels, **kwargs):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        if isinstance(batch, torch.Tensor): batch = batch.detach().cpu().numpy()
        img_anchors = kwargs['img_anchors']
        if isinstance(img_anchors, torch.Tensor): img_anchors = img_anchors.detach().cpu().numpy()
        label_set, count = np.unique(labels, return_counts=True)
        label_set = label_set[count >= 2]

        s_cross_anchors = []
        s_anchors = []

        bs = batch.shape[0]
        for i in range(bs):
            s_anchors.append(i)
            s_cross_anchors.append(labels[i])
        sample_pairs = [[a, p] for (a, p) in zip(s_anchors, s_cross_anchors)]

        pos_pairs = np.array([np.random.choice(np.where(labels == x)[0], 2, replace=False) for x in label_set])
        neg_tuples = []

        for idx in range(len(pos_pairs)):
            tuples = pos_pairs[np.delete(np.arange(len(pos_pairs)), idx), 1]
            p = pos_pairs[idx][0]
            cossim = []
            for n in tuples:
                v = batch[p] - img_anchors[idx]
                nv = batch[p] - batch[n]
                cossim.append(self.cos_sim(v, nv))
            cossim = np.stack(cossim)
            nidx = np.where(cossim < 0)
            n_tuples = tuples[nidx]
            if len(n_tuples) > 0:
                neg_tuples.append(n_tuples)

        npairs = [[a, *list(neg)] for (a, p), neg in zip(pos_pairs, neg_tuples)]

        return sample_pairs, npairs

    def anchor_npair(self, batch, labels, **kwargs):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        if isinstance(batch, torch.Tensor): batch = batch.detach().cpu().numpy()
        img_anchors = kwargs['img_anchors']
        if isinstance(img_anchors, torch.Tensor): img_anchors = img_anchors.detach().cpu().numpy()
        label_set, count = np.unique(labels, return_counts=True)
        label_set = label_set[count >= 2]

        s_cross_anchors = []
        s_anchors = []

        bs = batch.shape[0]
        for i in range(bs):
            s_anchors.append(i)
            s_cross_anchors.append(labels[i])
        sample_pairs = [[a, p] for (a, p) in zip(s_anchors, s_cross_anchors)]

        pos_pairs = np.array([np.random.choice(np.where(labels == x)[0], 2, replace=False) for x in label_set])
        neg_tuples = []

        for idx in range(len(pos_pairs)):

            neg_tuples.append(pos_pairs[np.delete(np.arange(len(pos_pairs)), idx), 1])

        neg_tuples = np.array(neg_tuples)

        npairs = [[a, p, *list(neg)] for (a, p), neg in zip(pos_pairs, neg_tuples)]
        return sample_pairs, npairs

    def anchor_npair_regression(self, batch, labels, **kwargs):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        if isinstance(batch, torch.Tensor): batch = batch.detach().cpu().numpy()
        img_anchors = kwargs['img_anchors']
        if isinstance(img_anchors, torch.Tensor): img_anchors = img_anchors.detach().cpu().numpy()
        label_set, count = np.unique(labels, return_counts=True)

        s_cross_anchors = []
        s_anchors = []

        bs = batch.shape[0]
        for i in range(bs):
            s_anchors.append(i)
            s_cross_anchors.append(labels[i])
        sample_pairs = [[a, p] for (a, p) in zip(s_anchors, s_cross_anchors)]

        pos_pairs = np.array([np.random.choice(np.where(labels == x)[0], 1, replace=False) for x in label_set])
        neg_tuples = []

        for idx in range(len(pos_pairs)):

            neg_tuples.append(pos_pairs[np.random.choice(np.delete(np.arange(len(pos_pairs)), idx), 1), 0])

        neg_tuples = np.array(neg_tuples)

        npairs = [[a, a, *list(neg)] for (a,), neg in zip(pos_pairs, neg_tuples)][0:1]
        return sample_pairs, npairs

    def pdist(self, A):
        """
        Efficient function to compute the distance matrix for a matrix A.

        Args:
            A:   Matrix/Tensor for which the distance matrix is to be computed.
            eps: float, minimal distance/clampling value to ensure no zero values.
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        return res.clamp(min=0).sqrt()

    def cos_sim(self, a, b):
        sim = np.dot(a, b) / (norm(a) * norm(b))
        return sim


class anchorNpairR(torch.nn.Module):
    def __init__(self, margin=1, sampling_method='anchor_npair', **kwargs):
        super(anchorNpairR, self).__init__()
        self.margin = margin
        self.sampler = TupleSampler(method=sampling_method)
        self.cross_weight = 1
        self.pair_weight = 1

    def positive_distance(self, anchor, corss_anchor):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.sqrt((anchor - corss_anchor).pow(2)).sum()

    def npair_distance(self, anchor, positive):
        """
        Compute basic N-Pair loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            n-pair loss (torch.Tensor())
        """
        return torch.nn.functional.relu(torch.sqrt((anchor - positive).pow(2)).sum() - self.margin)

    def weightsum(self, anchor, positive):
        """
        Compute weight penalty.
        NOTE: Only need to penalize anchor and positive since the negatives are created based on these.

        Args:
            anchor, positive: torch.Tensor(), resp. embeddings for anchor and positive samples.
        Returns:
            torch.Tensor(), Weight penalty
        """
        return torch.sum(anchor ** 2 + positive ** 2)

    def forward(self, batch, labels, img_anchor):
        # Sample triplets to use for training.
        samples = self.sampler.give(batch, labels, img_anchors=img_anchor)
        cross_pairs, positive_pairs = samples

        # Compute loss
        cross_loss = torch.mean(torch.stack(
            [self.positive_distance(batch[pair[0], :], img_anchor[pair[1], :]) for pair in cross_pairs]))

        npair_loss = torch.mean(torch.stack(
            [self.npair_distance(batch[pair[0], :], batch[pair[1], :]) for pair in positive_pairs]))

        if cross_loss < self.margin:
            loss = self.cross_weight * cross_loss + self.pair_weight * npair_loss
            print('Cross Loss: {:.5f}, Npair Loss: {:.5f}'.format(cross_loss.item(), npair_loss.item()))
        else:
            loss = self.cross_weight * cross_loss
            print('Cross Loss: {:.5f}'.format(cross_loss.item()))

        return loss


class AnchorLoss(torch.nn.Module):
    def __init__(self, sampling_method='positive_negative', **kwargs):
        super(AnchorLoss, self).__init__()
        self.sampler = TupleSampler(method=sampling_method)
        self.episode = 0

    def positive_distance(self, anchor, corss_anchor):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.sqrt((anchor - corss_anchor).pow(2)).sum()

    def forward(self, batch, labels, img_anchor):
        # Sample for training.
        samples = self.sampler.give(batch, labels, img_anchors=img_anchor)
        positive_pairs, negtive_pairs = samples

        sample_loss = torch.mean(torch.stack(
            [self.positive_distance(batch[pair[0], :], img_anchor[pair[1], :]) for pair in positive_pairs]))

        loss = sample_loss
        if self.episode == 0 or (self.episode + 1) % 10 == 0:
            print('Anchor Positive Loss: {:.5f}'.format(sample_loss.item()))

        self.episode += 1
        return loss


class CrossSample(torch.nn.Module):
    def __init__(self, margin=1, sampling_method='anchor_npair', **kwargs):
        super(CrossSample, self).__init__()
        self.margin = margin
        self.intra_margin = kwargs['intra']

        self.sampler = TupleSampler(method=sampling_method)
        self.num_classes = kwargs['num_classes']
        self.episode = 0

    def anchor_distance(self, anchor, cross_anchor):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.sqrt((anchor - cross_anchor).pow(2)).sum()

    def positive_distance(self, anchor, positive):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.nn.functional.relu(torch.sqrt((anchor - positive).pow(2)).sum() - self.intra_margin)

    def negative_distance(self, anchor, negatives):
        dist = torch.stack(
            [torch.nn.functional.relu(self.margin - torch.sqrt((anchor - neg).pow(2)).sum()) for neg in negatives])
        return torch.mean(dist)

    def forward(self, batch, labels, img_anchor):
        samples = self.sampler.give(batch, labels, img_anchors=img_anchor)
        anchor_pairs, npairs = samples

        anchor_loss = torch.mean(torch.stack(
            [self.anchor_distance(batch[pair[0], :], img_anchor[pair[1], :]) for pair in anchor_pairs]))

        negative_loss = torch.mean(torch.stack(
            [self.negative_distance(batch[pair[0], :], batch[pair[1:], :]) for pair in npairs]))

        loss = anchor_loss + negative_loss
        if self.episode == 0 or (self.episode + 1) % 10 == 0:
            print(
                'Anchor Positive Loss: {:.5f}, Sample Negative Loss: {:.5f}'.format(
                    anchor_loss.item(), negative_loss.item()))

        self.episode += 1
        return loss

# MMD
#
#
#
#
#
#
#
#
#
#
#
#
#         1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
#         1e3, 1e4, 1e5, 1e6
#     ]
#
#

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class MMD(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MMD, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.sampler = TupleSampler(method='anchor_npair')
        self.mmd_loss = MMDLoss().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.classifier = None

    def forward(self, batch, labels, img_anchor):
        if self.classifier is None:
            num_classes = torch.unique(labels).shape[0]
            self.classifier = nn.Linear(batch.shape[1], num_classes).to(batch.device)

        samples = self.sampler.give(batch, labels, img_anchors=img_anchor)
        anchor_pairs, npairs = samples

        target = torch.stack([batch[pair[0], :] for pair in anchor_pairs])
        source = torch.stack([img_anchor[pair[1], :] for pair in anchor_pairs])
        mmd_loss = self.mmd_loss(source, target)
        fea = self.classifier(batch)
        labels = labels.to(device=batch.device)
        class_loss = self.ce(fea, labels)
        loss = mmd_loss + class_loss
        return loss

# batch calculate batch anchor by label


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class PositiveNegativeLoss(torch.nn.Module):
    def __init__(self, margin=1, sampling_method='anchor_npair_regression', **kwargs):
        super(PositiveNegativeLoss, self).__init__()
        self.margin = margin
        self.intra_margin = kwargs['intra']
        self.sampler = TupleSampler(method=sampling_method)
        self.num_classes = kwargs['num_classes']
        self.episode = 0

    def positive_distance(self, anchor, corss_anchor):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.sqrt((anchor - corss_anchor).pow(2)).sum()

    def negative_distance(self, anchor, negatives):
        dist = torch.stack(
            [torch.sqrt((anchor - neg).pow(2)).sum() for neg in negatives])
        return torch.mean(dist)

    def weightsum(self, anchor, positive):
        """
        Compute weight penalty.
        NOTE: Only need to penalize anchor and positive since the negatives are created based on these.

        Args:
            anchor, positive: torch.Tensor(), resp. embeddings for anchor and positive samples.
        Returns:
            torch.Tensor(), Weight penalty
        """
        return torch.sum(anchor ** 2 + positive ** 2)

    def forward(self, batch, labels, img_anchor):
        # Sample triplets to use for training.

        samples = self.sampler.give(batch, labels, img_anchors=img_anchor)
        positive_pairs, negtive_pairs = samples

        sample_loss = torch.mean(torch.stack(
            [self.positive_distance(batch[pair[0], :], img_anchor[pair[1], :]) for pair in positive_pairs]))

        loss = sample_loss
        if self.episode == 0 or (self.episode + 1) % 10 == 0:
            print('Sample Positive Loss: {:.5f}'.format(sample_loss.item()))

        #     [self.negative_distance(batch[pair[0], :], batch[pair[1:], :]) for pair in negtive_pairs]))
        #
        #         'Sample Positive Loss: {:.5f}, Negative Loss: {:.5f}'.format(sample_loss.item(), negative_loss.item()))

        #         'Negative Loss: {:.5f}'.format(negative_loss.item()))

        self.episode += 1
        return loss

