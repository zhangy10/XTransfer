import numpy as np
from scipy import stats


def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        # start += (size // 2)
        start += size


def segment(x_train, y_train, num_channel, window_size=100):
    segments = np.zeros(((len(x_train) // (window_size // 2)), num_channel, window_size))
    labels = np.zeros(((len(y_train) // (window_size // 2))))
    index = 0
    for (start, end) in windowz(x_train, window_size):
        if (len(x_train[start:end]) == window_size):
            if (len(np.unique(y_train[start:end], return_counts=True)[0])) == 1:
                labels[index] = y_train[start]
                segments[index] = x_train[start:end].T
                index += 1
    return segments[:index, :, :], labels[:index]


def segment_more(x_train, y_train, num_channel, window_size=100):
    segments = np.zeros(((len(x_train) // (window_size // 2)), num_channel, window_size))
    labels = np.zeros(((len(y_train) // (window_size // 2))))
    index = 0
    for (start, end) in windowz(x_train, window_size):
        if (len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            labels[index] = m[0]
            segments[index] = x_train[start:end].T
            index += 1
    return segments[:index, :, :], labels[:index]


def map_label(except_class, label):
    except_class = np.asarray(except_class)
    n = np.argwhere(except_class < label)
    new_label = int(label - len(n))
    return new_label


if __name__ == "__main__":
    except_classes = [0, 1, 9, 10, 11, 12, 13, 14]
    a = map_label(except_classes, 3)
    print(a)
