import torch


def norm_of_kde(X, kernel_bandwidth):
    X_new = X.repeat(X.size(dim=0), 1, 1)
    X_new_tr = X.repeat(X.size(dim=0), 1, 1).transpose(0, 1)
    X_diff = X_new - X_new_tr
    return torch.exp(-(X_diff ** 2).sum(dim=2) / (2 * kernel_bandwidth * kernel_bandwidth)).mean()


def get_indices_with_lowest(arr, n):
    m = len(arr)
    assert n <= m

    list_ = []
    for i in range(m):
        list_.append((arr[i], i))

    list_.sort(key=lambda x: x[0])

    return [list_[i][1] for i in range(n)]