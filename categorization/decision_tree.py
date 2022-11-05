import csv
import sys
import numpy as np


class Tree:
    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.threshold = -1
        self.split_attr = -1
        self.depth = depth
        self.gini = -1
        self.values = None
        self.gini_gain = None

    def fit(self, ds, max_depth):
        self.values = ds
        self.left = None
        self.right = None
        # split classes from dataset
        classes = ds[:, -1].astype(dtype=int)
        ds_without_classes = ds[:, :-1]
        self.gini_gain = round(gini_impurity(ds, classes), 3)
        if self.gini_gain <= 0 or self.depth > max_depth:
            return
        # call gini
        gini, threshold, attr_idx = best_gini_split(ds_without_classes, classes)
        self.threshold = threshold
        self.split_attr = attr_idx
        self.gini = gini
        # make this tree a leaf node if perfect gini split

        self.left = Tree(self.depth + 1)
        self.left.fit(ds[ds[:, self.split_attr] <= self.threshold], max_depth)

        self.right = Tree(self.depth + 1)
        self.right.fit(ds[ds[:, self.split_attr] > self.threshold], max_depth)

        pass

    def validate():
        pass


def gini_impurity(ds, classes):
    class_set = set(classes)
    counts = {cs: (ds[:, -1] == cs).sum() for cs in class_set}
    probs = {cs: probability(count, len(classes)) for cs, count in counts.items()}
    return 1.0 - sum(probs.values())


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


def best_gini_split(ds, classes):
    best_attr_split = []
    for attr_idx, attr_values in enumerate(ds.T):
        mi = min(attr_values)
        ma = max(attr_values)
        # mi += 0.05
        # ma -= 0.05
        # print(mi, "\t", ma)
        gini_values = []

        for thold in np.arange(mi, ma, 0.05):
            gini_value = gini_split(attr_values, classes, thold)
            gini_values.append((round(gini_value, 4), round(thold, 4)))

        # print(attr_idx, gini_values)
        # the attributes values are so close together
        if not gini_values:
            # append gini for Minimum value and minimum value
            gini_values.append((gini_split(attr_values, classes, mi), mi))

        min_gini, min_thold = min(gini_values)
        best_attr_split.append((min_gini, min_thold, attr_idx))

    print(best_attr_split)
    min_gini, min_thold, best_split = min(best_attr_split)

    for g, t, s in best_attr_split:
        if g == min_gini:
            min_gini, min_thold, best_split = g, t, s
            break

    return min_gini, min_thold, best_split


if __name__ == "__main__":
    finput = sys.argv[1]
    ds = []

    with open(finput) as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if row:
                ds.append(row)
    ds = np.array(ds, dtype=float)
    root = Tree()
    root.fit(ds, 4)

    # print(ds.T)

    # tr = Tree()
    # tr.fit(ds)
