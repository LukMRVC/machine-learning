import csv
import sys
import numpy as np


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.excluded = set()
        self.threshold = -1
        self.split_arg = -1


class Tree:
    def __init__(self):
        self.root = None

    def fit(self, ds):
        dim = len(ds.T)
        for attr_values in ds.T:
            mi = min(attr_values)
            ma = max(attr_values)
            # print(attr_values)
            for thold in np.arange(mi, ma, 0.1):
                satisfied = attr_values < thold
                print(thold, ": ", satisfied.sum())
                prob = satisfied.sum()
                prob *= prob
                result = 1 - prob

        pass

    def validate():
        pass


def probability(count, total):
    if count <= 0 or total <= 0:
        return 0
    return (count / total) * (count / total)


def gini_split(values, classes, split):
    zipped = np.dstack([values, classes])[0]
    class_set = set(classes)

    left = zipped[zipped[:, 0] <= split]
    right = zipped[zipped[:, 0] > split]

    left_class_counts = {cs: (left[:, 1] == cs).sum() for cs in class_set}
    right_class_counts = {cs: (right[:, 1] == cs).sum() for cs in class_set}

    left_probabilities = {
        cs: probability(class_count, len(left))
        for cs, class_count in left_class_counts.items()
    }

    right_probabilities = {
        cs: probability(class_count, len(right))
        for cs, class_count in right_class_counts.items()
    }

    gini_left = 1 - sum(left_probabilities.values())
    gini_right = 1 - sum(right_probabilities.values())

    gini_total_part = lambda g, lr: g * len(lr) / len(values)

    gini_impurity = gini_total_part(gini_left, left) + gini_total_part(
        gini_right, right
    )

    return gini_impurity


def gini(ds, classes, excluded=[]):
    best_attr_split = []
    for attr_idx, attr_values in enumerate(ds.T[:-1]):
        if attr_idx in excluded:
            continue
        mi = min(attr_values)
        ma = max(attr_values)
        mi += 0.05
        ma -= 0.05
        # print(mi, "\t", ma)
        gini_values = []

        for thold in np.arange(mi, ma, 0.1):
            gini_value = gini_split(attr_values, classes, thold)
            gini_values.append((round(gini_value, 2), round(thold, 2)))

        min_gini, min_thold = min(gini_values)
        best_attr_split.append((min_gini, min_thold, attr_idx))
        print("Min: ", min(gini_values), attr_idx)

    print(best_attr_split)
    min_gini, min_thold, best_split = min(best_attr_split)
    print(f"Best split: {min_gini} threshold: {min_thold} on attribute {best_split}")

    # print("Impurity: ", impurity)


if __name__ == "__main__":
    finput = sys.argv[1]
    ds = []

    with open(finput) as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if row:
                ds.append(row)
    ds = np.array(ds, dtype=float)
    # split classes from dataset
    classes = ds[:, -1].astype(dtype=int)
    ds = ds[:, :-1]
    # call gini
    gini(ds, classes)
    # print(ds.T)

    # tr = Tree()
    # tr.fit(ds)
