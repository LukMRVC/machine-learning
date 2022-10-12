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
        self.subset_transactions = set()
        self.transactions = transactions
        self.depth = depth
        self._support_cache = None

    def iter_subset_trans(self):
        if self.parent is None:
            for idx, t in enumerate(self.transactions):
                yield t, idx
        else:
            for idx in self.subset_transactions:
                yield self.transactions[idx], idx

    def __repr__(self):
        return f'D:{self.depth} -> V:{self.values} (supp={self.get_support()})'

    def get_support(self) -> float:
        if self._support_cache is not None:
            return self._support_cache

        if self.values == set():
            return 1

        in_transactions = 0
        missing = (self.values - self.parent.values).pop()

        for t, tran_idx in self.parent.iter_subset_trans():
            if t[missing- 1] == 1:
                in_transactions += 1
                self.subset_transactions.add(tran_idx)
        supp = in_transactions / len(self.transactions)
        self._support_cache = supp
        return supp

    def get_child_support(self, child_values: set[int]) -> float:
        if len(self.transactions) <= 0:
            return 1, []

        in_transactions = 0
        missing = (child_values - self.values).pop()
        child_transactions = set()
        for t, tran_idx in self.iter_subset_trans():
            if t[missing - 1] == 1:
                in_transactions += 1
                child_transactions.add(tran_idx)
        
        return in_transactions / len(self.transactions), child_transactions

    def find_itemset_support(self, itemset: set[int], traverse_up = True):
        if self.parent is not None and traverse_up:
            return self.parent.find_itemset_support(itemset)

        for child in self.children:
            if child.values.issubset(itemset):
                if len(child.values) == len(itemset):
                    return child.get_support()
                else:
                    return child.find_itemset_support(itemset, False)


    def get_conf(self, left_side: set[int]) -> float:
        total_sup = self.get_support()
        left_sup = self.parent.find_itemset_support(left_side)
        if left_sup > 0:
            return total_sup / left_sup
        return 0

    def generate_combs(self, max_depth: int = -1, min_sup: float = 0, min_conf: float = 0):
        if self.depth >= max_depth != -1:
            return

        if self.depth == 0 and len(self.children) > 0:
            self.children.clear()

        mn = min(self.items) - 1 if self.values == set() else max(self.values)
        itemset_candidates = [it for it in self.items if it > mn]
        for it in itemset_candidates:
            # prune the tree -> dont appen children, whose support is less than min_sup
            # each superset will only have smaller support
            sup, child_transactions = self.get_child_support({*self.values, it})
            if min_sup > 0:
                # if os.environ.get('DEBUG', '0') == '1':
                #     print(f'Calculating support in dephth: {self.depth}')
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
            self.children[-1]._support_cache = sup
            self.children[-1].subset_transactions = child_transactions
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
        print(f'{self.values} (supp={self.get_support()})')
        for c in self.children:
            c.print_tree()


def normalize_transactions(transactions: list[list[int]], items: set[int]) -> list[list[int]]:
    for idx, t in enumerate(transactions):
        normalized_transaction = [0] * max(items)
        for it in t:
            normalized_transaction[it - 1] = 1
        transactions[idx] = normalized_transaction
    return transactions


def read_transactions(filepath: str) -> tuple[list[list[int]], set[int]]:
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

    print('--------- TEST DATA TREE (minsup=0.25) ----------------')
    transactions, items = read_transactions('./itemsets_test.dat')
    transactions = normalize_transactions(transactions, items)
    t = RaymonTree(cols=len(items), transactions=transactions)
    # generate tree with min_support of 0.25
    t.root_node.generate_combs(min_sup=0.25)
    t.root_node.print_tree()
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

    transactions, items = read_transactions('./connect-smaller.dat')
    transactions = normalize_transactions(transactions, items)
    print('Transactions normalized, generating raymon tree')
    cols = len(items)
    print(f'Cols = {cols}, num. of combinations = {2**cols}')
    t = RaymonTree(cols=len(items), transactions=transactions)
    # generate tree with min_support of 0.25
    t.root_node.generate_combs(min_sup=0.25)
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
