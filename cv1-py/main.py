import os
from itertools import combinations
from enum import Enum
from dataclasses import dataclass


class ReturnType(str, Enum):
    support = 'support'
    confidence = 'confidence'


class RaymonTree:
    def __init__(self, cols, transactions: list[list[int]]):
        self.items: set[int] = set(range(1, cols + 1))
        self.transactions: list[list[int]] = transactions
        self.root_node = TreeNode(set(), None, 0, self.items, self.transactions)


@dataclass
class AssociationRule:
    left_side: tuple[int]
    right_side: int
    confidence: float

    def __repr__(self):
        return f'{",".join(map(str, self.left_side))} -> {self.right_side} (conf={round(self.confidence, 2)})'


class TreeNode:
    def __init__(self, value: set[int], parent, depth: int, items: set[int], transactions: list[list[int]]):
        self.values: set[int] = value
        self.parent = parent
        self.children: list[TreeNode] = []
        self.items = items
        self.transactions = transactions
        self.depth = depth
        self._support_cache = None

    def __repr__(self):
        return f'D:{self.depth} -> V:{self.values}'

    def get_support(self, values: set[int] = None) -> float:
        if self._support_cache is not None and values is None:
            return self._support_cache

        if self.values == set():
            return 1

        in_transactions = 0

        for t in self.transactions:
            for it in (values or self.values):
                if t[it - 1] == 0:
                    break
            else:
                in_transactions += 1
        supp = in_transactions / len(self.transactions)
        if values is None:
            self._support_cache = supp
        return supp

    def get_conf(self, left_side: set[int]) -> float:
        total_sup = self.get_support()
        left_sup = self.get_support(left_side)
        if left_sup > 0:
            return total_sup / left_sup
        return 0

    def generate_combs(self, max_depth: int = -1, min_sup: float = 0, min_conf: float = 0):
        if self.depth >= max_depth != -1:
            return

        if self.depth == 0 and len(self.children) > 0:
            self.children.clear()

        mn = min(self.items) - 1 if self.values == set() else max(self.values)
        for it in [it for it in self.items if it > mn]:
            # prune the tree -> dont appen children, whose support is less than min_sup
            # each superset will only have smaller support
            if min_sup > 0:
                # if os.environ.get('DEBUG', '0') == '1':
                #     print(f'Calculating support in dephth: {self.depth}')
                sup = self.get_support({*self.values, it})
                if sup < min_sup:
                    continue

            self.children.append(
                TreeNode(
                    {*self.values, it},
                    self,
                    self.depth + 1,
                    self.items,
                    self.transactions,
                )
            )
            self.children[-1].generate_combs(max_depth, min_sup, min_conf)

    def get_rules(self, min_sup: float = 0, min_conf: float = 0) -> list[list[int]]:
        return_list = []
        if self.depth != 0 and min_sup > 0:
            sup = self.get_support()
            if sup >= min_sup:
                return_list.append(self.values)

        if self.depth > 1 and min_conf > 0:
            for left_side in combinations(list(self.values), len(self.values) - 1):
                mnum = 0
                for v in self.values:
                    mnum ^= v
                for l in left_side:
                    mnum ^= l
                conf = self.get_conf(set(left_side))
                if conf >= min_conf:
                    return_list.append(AssociationRule(
                        left_side=left_side,
                        right_side=mnum,
                        confidence=conf
                    ))

        for c in self.children:
            return_list.extend(c.get_rules(min_sup, min_conf))

        return return_list

    def print_tree(self):
        print(self.values)
        for c in self.children:
            c.print_tree()


def normalize_transactions(transactions: list[list[int]], items: set[int]) -> list[list[int]]:
    for idx, t in enumerate(transactions):
        normalized_transaction = [0] * len(items)
        for it in t:
            normalized_transaction[it - 1] = 1
        transactions[idx] = normalized_transaction
    return transactions


def read_transactions(filepath: str) -> (list[list[int]], set[int]):
    transactions = []
    items = set()

    with open(filepath) as f:
        for line in f:
            transactions.append(list(map(int, line.strip().split(' '))))
            items = items.union(set(transactions[-1]))

    return transactions, items


if __name__ == '__main__':
    t = RaymonTree(cols=6, transactions=[])
    t.root_node.generate_combs(3)
    print('Printing all combinations in tree of length 3')
    t.root_node.print_tree()

    transactions, items = read_transactions('./itemsets_test.dat')
    transactions = normalize_transactions(transactions, items)
    t = RaymonTree(cols=len(items), transactions=transactions)
    # generate tree with min_support of 0.25
    t.root_node.generate_combs(min_sup=0.25)
    # get all rules with min_supp of 0.25
    support_rules = t.root_node.get_rules(min_sup=0.25)
    print('Patterns with min_supp >= 0.25: ', support_rules)
    # get all rules with min_confidence of 0.5
    conf_rules = t.root_node.get_rules(min_conf=0.5)
    print('Association rules with min_conf >= 0.5: ')
    for rule in conf_rules:
        print(rule)

    # generate tree again with min_support of 0.15
    t.root_node.generate_combs(min_sup=0.15)
    # get all rules with min_supp of 0.15
    support_rules = t.root_node.get_rules(min_sup=0.15)
    print('Patterns with min_supp >= 0.15: ', support_rules)
    # get all rules with min_confidence of 0.5
    conf_rules = t.root_node.get_rules(min_conf=0.5)
    print('Association rules with min_conf >= 0.5: ')
    for rule in conf_rules:
        print(rule)

    print('------------------ CONNECT DATA FILE ------------------------')

    transactions, items = read_transactions('./connect.dat')
    transactions = normalize_transactions(transactions, items)
    print('Transactions normalized, generating raymon tree')
    cols = len(items)
    print(f'Cols = {cols}, num. of combinations = {2**cols}')
    t = RaymonTree(cols=len(items), transactions=transactions)
    # generate tree with min_support of 0.25
    t.root_node.generate_combs(min_sup=0.33)
    # get all rules with min_supp of 0.33
    support_rules = t.root_node.get_rules(min_sup=0.33)
    print('Patterns with min_supp >= 0.33: ', support_rules)
    # get all rules with min_confidence of 0.5
    conf_rules = t.root_node.get_rules(min_conf=0.5)
    print('Association rules with min_conf >= 0.5: ')
    for rule in conf_rules:
        print(rule)

    # generate tree again with min_support of 0.15
    t.root_node.generate_combs(max_depth=5, min_sup=0.15)
    # get all rules with min_supp of 0.15
    support_rules = t.root_node.get_rules(min_sup=0.15)
    print('Patterns with min_supp >= 0.15: ', support_rules)
    # get all rules with min_confidence of 0.5
    conf_rules = t.root_node.get_rules(min_conf=0.5)
    print('Association rules with min_conf >= 0.5: ')
    for rule in conf_rules:
        print(rule)
