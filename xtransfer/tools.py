from sklearn.metrics.pairwise import paired_distances, pairwise_distances_chunked
from sklearn.decomposition import PCA
# from sklearn.decomposition import KernelPCA
# from sklearn.decomposition import SparsePCA as PCA

from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
import numpy as np
import hashlib
import torch
import torch.nn.functional as F
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import functools
import torch.nn
from sklearn.metrics.pairwise import euclidean_distances
# import pyemd
from utils import Notes

best_pca = {}


def bn_modification(model, momentum=0.1, is_train=True):
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.modules.BatchNorm2d):
            layer.momentum = momentum
            # layer.running_mean = None
            # layer.running_var = None
            # layer.running_mean = layer.running_mean.clone().fill_(0)
            # layer.running_var = layer.running_var.clone().fill_(1)
            layer.train(is_train)
    return model


def get_next_key(dict, key, skip_num=1):
    temp = list(dict)
    try:
        res = temp[temp.index(key) + skip_num]
    except (ValueError, IndexError):
        res = None
    return res


def build_model_dict(m):
    dict = {}
    for name, item in m.named_modules():
        dict[name] = item
    return dict


class PCAO(PCA):
    def __init__(self, n_components=0.8):
        super(PCAO, self).__init__(n_components=n_components)

    def transform_s(self, X):
        if not isinstance(X, np.ndarray):
            X = X.numpy()
        X -= np.mean(X, axis=0)
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed


def silhouette_reduce_our(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X.

    Parameters
    ----------
    D_chunk : array-like of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk.
    start : int
        First index in the chunk.
    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)), dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(
            labels, weights=D_chunk[i], minlength=len(label_freqs)
        )

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start: start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inner_clust_dists = clust_dists.min(axis=1)
    inter_clust_dists = []
    for cd in clust_dists:
        mean = np.mean(cd[~np.isinf(cd)])
        inter_clust_dists.append(mean)
    inter_clust_dists = np.stack(inter_clust_dists)
    return inner_clust_dists, inter_clust_dists


def intra_distance(X, labels, metric="euclidean", **kwds):
    X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    kwds["metric"] = metric
    reduce_func = functools.partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs
    )
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, _ = results
    intra_clust_dists = np.concatenate(intra_clust_dists)

    denom = (label_freqs - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom
    return mean_distance(np.nan_to_num(intra_clust_dists), labels)


def inter_distance(X, labels, metric="euclidean", **kwds):
    X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    kwds["metric"] = metric
    reduce_func = functools.partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs
    )
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    _, inter_clust_dists = results
    inter_clust_dists = np.concatenate(inter_clust_dists)
    return mean_distance(np.nan_to_num(inter_clust_dists), labels)


def hash_tensor(tensor):
    # check if the tensor is contiguous (required for hashing)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    # generate a hash value using the tensor data and shape
    hash_value = hashlib.sha256(tensor.numpy().tobytes() + torch.tensor(tensor.shape).numpy().tobytes()).hexdigest()
    return hash_value


def pca_fit(samples, n_comp=0.8, **kwargs):
    if n_comp == 'best':
        hash_value = hash_tensor(samples)
        if not hash_value in best_pca.keys():
            labels = kwargs['labels']
            best_n, sscore = find_bestN(samples, labels)
            best_pca[hash_value] = best_n
            print('Find best #Components: {}'.format(best_n))
            Notes.write('Find best #Components: {}'.format(best_n))
        else:
            best_n = best_pca[hash_value]
        pca = PCA(n_components=best_n, random_state=1)
        # pca = PCA(n_components=best_n, random_state=1, kernel='linear')
        pca_fitted = pca.fit(samples)
    else:
        if n_comp < 1.0:
            n_comp = int(min(samples.shape[0], samples.shape[1]) * n_comp)
        pca = PCA(n_components=n_comp, random_state=1)
        # pca = PCA(n_components=n_comp, random_state=1, kernel='linear')
        # pca = PCA(n_components=n_comp, random_state=1, alpha=0.1)
        pca_fitted = pca.fit(samples)
    return pca_fitted


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # 数组转列表
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def mmc(feature):
    if not torch.is_tensor(feature):
        feature = torch.from_numpy(feature)

    if feature.dim() > 4:
        shape = list([-1] + list(feature.shape[2:]))
        feature = feature.view(shape)

    # dim is 2, which means the tensor is mmc.
    if feature.dim() == 2:
        return feature

    # l2 norm to feature
    if feature.dim() == 3:
        feature = F.adaptive_avg_pool1d(feature, 1).squeeze_(-1)
    else:
        feature = F.adaptive_avg_pool2d(feature, 1).squeeze_(-1).squeeze_(-1)
    return feature


# def concat_feature(features, labels):
#     if not isinstance(features, np.ndarray):
#         features = features.numpy()
#         labels = labels.numpy()
#     fea_dim = features.shape[-1]
#     feature_per_class = np.zeros((len(np.unique(labels)), fea_dim), dtype=np.float32)
#     weight = np.zeros((len(np.unique(labels)),), dtype=np.float32)
#     dic = {}
#     for i, c in enumerate(np.unique(labels)):
#         cmask = (labels == c)
#         feature_per_class[i, :] = np.mean(features[cmask, :], axis=0)
#         weight[i] = np.sum(cmask)
#     dic['feature'] = feature_per_class
#     dic['weight'] = weight
#     return dic


# def domain_similarity(x, y, anchor_x, anchor_y, gamma=0.01):
#     target_dic = concat_feature(x, y)
#     source_dic = concat_feature(anchor_x, anchor_y)
#     f_s = source_dic['feature']
#     w_s = source_dic['weight']
#     f_t = target_dic['feature']
#     w_t = target_dic['weight']
#     data = np.float64(np.append(f_s, f_t, axis=0))
#     w_1 = np.zeros((len(w_s) + len(w_t),), np.float64)
#     w_2 = np.zeros((len(w_s) + len(w_t),), np.float64)
#     w_1[:len(w_s)] = w_s / np.sum(w_s)
#     w_2[len(w_s):] = w_t / np.sum(w_t)
#     dist = euclidean_distances(data, data)
#     emd = pyemd.emd(np.float64(w_1), np.float64(w_2), np.float64(dist))
#     similarity = np.exp(-gamma * emd)
#     return dist, emd, similarity


def get_mean(data):
    mean = np.mean(data, axis=0, keepdims=False)
    return mean


def get_medoid(data):
    # get mean
    mean = np.mean(data, axis=0, keepdims=True)
    means = np.asarray(np.repeat(mean, len(data), axis=0))
    idx = np.argmin(paired_distances(data, means))
    medoid = data[idx]
    return medoid


def get_medoid_id(data):
    # get mean
    mean = np.mean(data, axis=0, keepdims=True)
    means = np.asarray(np.repeat(mean, len(data), axis=0))
    idx = np.argmin(paired_distances(data, means))
    return idx


def class_medoids(data):
    medoids = []
    for d in data:
        m = get_medoid(d)
        medoids.append(m)
    return np.asarray(medoids)


def find_closet_anchor(target, target_y, anchor, anchor_y):
    target_medoids = class_medoids(target, target_y)
    anchor_medoids = class_medoids(anchor, anchor_y)
    anchors, anchor_ys = map_classes_closet(anchor_medoids, target_medoids)
    return anchors, anchor_ys


def class_medoids(data, label):
    data = reshape_dim2(data)
    medoids = []
    num_classes = len(np.unique(label))
    for c in range(num_classes):
        d = data[label == c]
        m = get_medoid(d)
        medoids.append(m)
    return np.asarray(medoids)


def class_centroids(data, label):
    data = reshape_dim2(data)
    centroids = []
    for c in np.unique(label):
        d = data[label == c]
        m = d.mean(axis=0)
        centroids.append(m)
    return np.asarray(centroids)


def class_centroids_dict(data, label):
    data = reshape_dim2(data)
    centroids = {}
    for c in np.unique(label):
        d = data[label == c]
        m = d.mean(axis=0).numpy()
        centroids[c] = m
    return centroids


def class_centroids_sort(data, label):
    data = reshape_dim2(data)
    centroids = []
    for c in range(len(np.unique(label))):
        d = data[label == c]
        m = d.mean(axis=0)
        centroids.append(m)
    return torch.stack(centroids)


#
# def class_centroids_best(data, label):
#     data = reshape_dim2(data)
#     centroids = []
#     labels = []
#     for c in np.unique(label):
#         d = data[label == c]
#         m = d.mean(axis=0)
#         centroids.append(m)
#         labels.append(c)
#     return np.asarray(centroids), np.asarray(labels)


def mean_distance(data, label):
    means = []
    num_classes = len(np.unique(label))
    for c in range(num_classes):
        d = data[label == c]
        m = d.mean(axis=0)
        means.append(m)
    return np.asarray(means)


def class_medoids_id(data, label):
    data = reshape_dim2(data)
    medoids = []
    num_classes = len(np.unique(label))
    for c in range(num_classes):
        d = data[label == c]
        mid = get_medoid_id(d)
        idx = np.where(label == c)[0][mid]
        medoids.append(idx)
    return np.asarray(medoids)


def reshape_dim2(data):
    if data.ndim > 2:
        data = data.reshape((data.shape[0], -1))
    return data


def select_topN(anchor_map, top_n=7):
    if not isinstance(anchor_map, dict):
        anchor_map = {key: anchor_map[key] for key in range(len(anchor_map))}

    sorted_x = sorted(anchor_map.items(), key=lambda kv: kv[1], reverse=True)
    topN = np.asarray(sorted_x)[:top_n, 0].astype(int)

    return topN


def map_classes(anchor_medoids, anchor_topN, target_medoids, target_topN):
    # map  {target:anchor, ...}
    class_map = {}

    for t in target_topN:
        tm = target_medoids[t]
        tms = np.asarray([tm for _ in range(len(anchor_topN))])
        ams = anchor_medoids[anchor_topN]
        idx = np.argmin(paired_distances(tms, ams))
        class_map[t] = anchor_topN[idx]
        anchor_topN = np.delete(anchor_topN, idx)

    return class_map


def map_classes_best(anchor_medoids, anchor_topN, target_medoids, target_topN):
    """
    Args:
        anchor_medoids:
        anchor_topN:
        target_medoids:
        target_topN:
    References: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
    """

    class_map = {}
    best_anchors = anchor_medoids[anchor_topN]
    # best_anchors = anchor_medoids
    num_classes = len(target_medoids)
    coords = distance.cdist(best_anchors, target_medoids, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(coords)


    for rid, cid in zip(row_ind, col_ind):
        class_map[cid] = anchor_topN[rid]
    dist = coords[row_ind, col_ind].sum()
    print('Best anchor matching is {}, Sum of distance is {:.4f}'.format(class_map, dist))

    return class_map, dist


def find_bestN(data, label):
    max_n = min(data.size(1) + 1, data.size(0) + 1)
    ss = {}
    for n in range(1, max_n):
        fitted_pca = pca_fit(data, n_comp=n)
        data_pca = fitted_pca.transform(data)
        score = class_silhouette_score(data_pca, label)
        score = np.mean([v for _, v in score.items()])
        ss[n] = score
    topn = select_topN(ss, top_n=1)
    sscore = ss[topn[0]]
    return topn[0], sscore


def map_classes_closet(anchor_medoids, target_medoids):
    # map  {target:anchor, ...}
    anchor = []
    anchor_y = np.asarray([c for c in range(len(anchor_medoids))])
    ys = []
    for tm in target_medoids:
        tms = np.asarray([tm for _ in range(len(anchor_medoids))])
        # ams = anchor_medoids
        idx = np.argmin(paired_distances(tms, anchor_medoids))
        anchor.append(anchor_medoids[idx])
        ys.append(anchor_y[idx])
        anchor_medoids = np.delete(anchor_medoids, idx, 0)
        anchor_y = np.delete(anchor_y, idx, 0)
    anchor = np.stack(anchor)
    ys = np.stack(ys)
    return anchor, ys


def generate_labels(data):
    labels = np.asarray([i for i in range(data.shape[0]) for _ in range(data.shape[1])])
    return labels


def class_silhouette_score(pca_data, labels, precision=None, regression=False):
    if regression:
        sample_silhouette_values = silhouette_samples_ours(pca_data, labels)
    else:
        sample_silhouette_values = silhouette_samples(pca_data, labels)
    mm = {}
    for i in np.unique(labels):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        mean_value = np.mean(ith_cluster_silhouette_values)
        if precision:
            mm[i] = np.round(mean_value, precision)
        else:
            mm[i] = mean_value
    return mm


def silhouette_samples_ours(X, labels, *, metric="euclidean", **kwds):
    """Compute the Silhouette Coefficient for each sample.

    The Silhouette Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 ``<= n_labels <= n_samples - 1``.

    This function returns the Silhouette Coefficient for each sample.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.

    Read more in the :ref:`User Guide <silhouette_coefficient>`.

    Parameters
    ----------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
        Label values for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.

    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    silhouette : array-like of shape (n_samples,)
        Silhouette Coefficients for each sample.

    References
    ----------

    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    """
    X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])

    # Check for non-zero diagonal entries in precomputed distance matrix
    if metric == "precomputed":
        atol = np.finfo(X.dtype).eps * 100
        if np.any(np.abs(np.diagonal(X)) > atol):
            raise ValueError(
                "The precomputed distance matrix contains non-zero "
                "elements on the diagonal. Use np.fill_diagonal(X, 0)."
            )

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    kwds["metric"] = metric
    reduce_func = functools.partial(
        silhouette_reduce_our, labels=labels, label_freqs=label_freqs
    )
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists) * 1 / 2
    inter_clust_dists = np.concatenate(inter_clust_dists)

    # denom = (label_freqs - 1).take(labels, mode="clip")
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def silhouette_score(pca_data, labels):
    sample_silhouette_values = silhouette_samples(pca_data, labels)
    score = np.mean(sample_silhouette_values)
    return score


def map_anchor_target(af, tf, al, tl):
    af = reshape_dim2(af)
    tf = reshape_dim2(tf)

    aMedoids = class_medoids(af, al)
    tMedoids = class_medoids(tf, tl)
    topN = len(np.unique(tl))

    aScore = class_silhouette_score(af, al)
    tScore = class_silhouette_score(tf, tl)

    aTOPN = select_topN(aScore, topN)
    tTOPN = select_topN(tScore, topN)

    class_map = map_classes(aMedoids, aTOPN, tMedoids, tTOPN)
    return class_map, aScore, tScore


def map_anchor_target_pca(af, tf, al, tl, ncomp=0.8, **kwargs):
    af = reshape_dim2(af)
    tf = reshape_dim2(tf)

    # fitting pca and transform af and tf
    # fitted_pca = pca_fit_o(af, n_comp=ncomp)
    # af = fitted_pca.transform(af)
    # tf = fitted_pca.transform_s(tf)
    if 'fitted_pca' in kwargs:
        fitted_pca = kwargs['fitted_pca']
    else:
        fitted_pca = pca_fit(af, n_comp=ncomp)
    af = fitted_pca.transform(af)
    tf = fitted_pca.transform(tf)

    aMedoids = class_medoids(af, al)
    tMedoids = class_medoids(tf, tl)
    topN = len(np.unique(tl))

    aScore = class_silhouette_score(af, al)
    tScore = class_silhouette_score(tf, tl)

    aTOPN = select_topN(aScore, topN)
    tTOPN = select_topN(tScore, topN)

    class_map = map_classes(aMedoids, aTOPN, tMedoids, tTOPN)
    return class_map, aScore, tScore, fitted_pca


# def map_anchor_target_pca_sensing(af, tf, al, tl, ncomp=0.8):
#     af = reshape_dim2(af)
#     tf = reshape_dim2(tf)
#
#     # fitting pca and transform af and tf
#     # fitted_pca = pca_fit_o(af, n_comp=ncomp)
#     # af = fitted_pca.transform(af)
#     # tf = fitted_pca.transform_s(tf)
#
#     fitted_pca = pca_fit(tf, n_comp=ncomp)
#     af = fitted_pca.transform(af)
#     tf = fitted_pca.transform(tf)
#
#     aMedoids = class_medoids(af, al)
#     tMedoids = class_medoids(tf, tl)
#     topN = len(np.unique(tl))
#
#     aScore = class_silhouette_score(af, al)
#     tScore = class_silhouette_score(tf, tl)
#
#     aTOPN = select_topN(aScore, topN)
#     tTOPN = select_topN(tScore, topN)
#
#     class_map = map_classes(aMedoids, aTOPN, tMedoids, tTOPN)
#     return class_map, aScore, tScore, fitted_pca


def class_accuracy(y_true, y_pred):
    cmat = confusion_matrix(y_true, y_pred)
    class_acc = cmat.diagonal() / cmat.sum(axis=1)
    return class_acc


def replace_relu(model):
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False


def align_anchor_to_target(n_shot, dict):
    aligned_anchor = {}
    label = dict['label']
    class_dict = {}
    for i in np.unique(label):
        idx = np.argwhere(label == i).squeeze()
        class_dict[i] = idx

    for key, item in dict.items():
        item = item
        all_means = []
        for cid, idx in class_dict.items():
            means = np.stack([np.mean(d, axis=0) for d in np.array_split(item[idx], n_shot)])
            all_means.append(means)
        if key == 'label':
            aligned_anchor[key] = torch.from_numpy(np.concatenate(all_means, axis=0)).long()
        else:
            aligned_anchor[key] = torch.from_numpy(np.concatenate(all_means, axis=0))
    return aligned_anchor


if __name__ == "__main__":
    from utils import load_dict, save_dict

    # train_x = load_dict('../datasets/data/face/og_bn_noneval_c-1_train.pkl')[19]
    #
    # train_y = torch.from_numpy(generate_labels(np.zeros((7, 5))))
    # test_x = load_dict('../datasets/data/face/og_bn_noneval_c-1_test.pkl')[19]
    # test_y = torch.from_numpy(generate_labels(np.zeros((7, 50))))
    # train_x = reshape_dim2(train_x)
    # distance = intra_distance(train_x, train_y)
    # print()
    anchor_x = load_dict('../datasets/data/mini-imagenet/mean-split_bn_eval_c-1.pkl')
    anchor_y = generate_labels(np.zeros((100, 10)))
    train_x = load_dict('../datasets/data/face/og_bn_noneval_c-1_train.pkl')
    train_y = generate_labels(np.zeros((7, 5)))
    test_x = load_dict('../datasets/data/face/og_bn_noneval_c-1_test.pkl')
    test_y = generate_labels(np.zeros((7, 50)))

    saved_layers = (0, 4, 9, 14, 19)
    ii = 0
    num_classes = 7
    for layer_id in saved_layers:
        print("L{}:".format(ii))
        ax = anchor_x[layer_id]
        ax = ax.reshape(([-1] + list(ax.shape[2:])))
        # ax_at = copy.deepcopy(ax)
        ax = mmc(ax)

        tx = train_x[layer_id]
        # tx = normalization(tx)
        tx = mmc(tx)

        class_map, aScore, tScore, fitted_pca = map_anchor_target_pca(ax, tx, anchor_y, train_y)
        print()

