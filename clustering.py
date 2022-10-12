from __future__ import annotations
from iris_data_matrix import load_iris_dataset
from karate_club import Matrix
from enum import Enum


class ClusterDistance(Enum):
    SINGLE_LINK = 0
    COMPLETE_LINK = 1
    AVERAGE_LINK = 2


def euclid_distance(a: iter, b: iter) -> float:
    distance = 0
    for idx in range(len(a)):
        point_distance = (a[idx] - b[idx]) * (a[idx] - b[idx])
        distance += point_distance
    return distance


class Cluster:
    def __init__(self, id: int, members: list):
        self.id = id
        self._members = members.copy()
        self.history = [id]

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return f'Cluster ID {self.id} of {len(self._members)}: {str(tuple(self._members))}'

    def __eq__(self, other):
        if isinstance(other, Cluster):
            return self._members == other.members

    @property
    def members(self) -> list:
        return self._members

    def append(self, cluster: Cluster):
        self.history.extend(cluster.history)
        self._members.extend(cluster.members)

    def calc_distance(self, cluster: Cluster, distance_calc_method: ClusterDistance) -> float:
        distances = []
        for m in self._members:
            distances.extend(cluster._distance_from(m))

        if distance_calc_method == ClusterDistance.SINGLE_LINK:
            return min(distances)
        else:
            raise RuntimeError("Distance calc method not implemented")

    def _distance_from(self, vector) -> list:
        distances = []
        for m in self._members:
            distances.append(round(euclid_distance(vector, m), 3))
        return distances


class DistanceMatrix():
    def __init__(self, size: int, init_with: int = 0):
        self.size = size
        self.m = [init_with] * (self.size * self.size)
        self.row_labels = {}

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        str_rep = ''
        for i, row in enumerate(self.rows_i()):
            str_rep += f'{self.row_labels[i]}: {row}\n'
        return str_rep

    def row_label(self, row, label):
        self.row_labels[row] = label

    def at(self, row, col):
        return self.m[row * self.size + col]

    def insert_at(self, row, col, val) -> None:
        self.m[row * self.size + col] = val
        self.m[col * self.size + row] = val

    def remove(self, row_col_idx):
        removed = 0
        for i in range(self.size):
            to_remove = (i * self.size) + row_col_idx - removed
            self.m.pop(to_remove)
            removed += 1
        b = self.m[:row_col_idx * (self.size - 1)]
        a = self.m[(row_col_idx + 1) * (self.size - 1):]
        b.extend(a)
        self.m = b.copy()
        self.size -= 1
        pass

    def row(self, idx: int) -> list:
        return self.m[idx * self.size: (idx + 1) * self.size]

    def rows_i(self):
        for i in range(0, len(self.m), self.size):
            yield self.m[i: i + self.size]


# pick_min_in_row
def pick_mir(row: list, row_idx: int) -> float:
    r = [x for i, x in enumerate(row) if i != row_idx]
    return min(r)
    pass


if __name__ == '__main__':
    iris_data = load_iris_dataset('iris.csv')
    clusters = [Cluster(i, [x]) for i, x in enumerate(iris_data)]
    dst_matrix = DistanceMatrix(size=len(clusters), init_with=-1)
    inserts = False
    for row, c1 in enumerate(clusters):
        dst_matrix.row_label(row, c1.id)
        for col, c2 in enumerate(clusters):
            # already calculated
            if dst_matrix.at(row, col) > -1:
                continue
            # zero the diagonal
            if c2 == c1:
                dst_matrix.insert_at(row, col, 0)
            else:
                dst_matrix.insert_at(row, col, c1.calc_distance(
                    c2, ClusterDistance.SINGLE_LINK))
                inserts = True

    it = 0
    while dst_matrix.size > 1:
        print(f'Iteration: {it}')
        it += 1
        rows_minimum = [pick_mir(r, i)
                        for i, r in enumerate(dst_matrix.rows_i())]
        cur_min = min(rows_minimum)
        row_idx = rows_minimum.index(cur_min)
        col_idx = dst_matrix.row(row_idx).index(cur_min)

        # print(row_idx, col_idx)
        c1 = clusters[row_idx]
        c2 = clusters[col_idx]
        print(f'{c1.history} -> {c2.history}')

        c1.append(c2)
        clusters.pop(col_idx)
        dst_matrix.remove(col_idx)

        for col, cluster in enumerate(clusters):
            if cluster == c1:
                dst_matrix.insert_at(row_idx, col, 0)
            else:
                dst_matrix.insert_at(row_idx, col, c1.calc_distance(
                    cluster, ClusterDistance.SINGLE_LINK))
        # print('----------------------------------')

    print(dst_matrix)
    print(clusters[0].history)
    pass
