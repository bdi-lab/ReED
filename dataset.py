import os
from tqdm import tqdm

class Data():
    def __init__(self, path):
        self.path = path
        self.ent2id, self.id2ent = self.read_items(os.path.join(path, "entity2id.txt"))
        self.rel2id, self.id2rel = self.read_items(os.path.join(path, "relation2id.txt"))
        self.train_pos = self.read_triplets(os.path.join(path, "train_pos.txt"))
        self.valid_pos = self.read_triplets(os.path.join(path, "valid_pos.txt"))
        self.test_pos = self.read_triplets(os.path.join(path, "test_pos.txt"))
        self.train_neg = self.read_triplets(os.path.join(path, "train_neg.txt"))
        self.valid_neg = self.read_triplets(os.path.join(path, "valid_neg.txt"))
        self.test_neg = self.read_triplets(os.path.join(path, "test_neg.txt"))

        self.check_dups()

    def read_items(self, path):
        item2id = {}
        id2item = []
        with open(path, "r") as f:
            for idx, line in enumerate(f.readlines()):
                item = line.strip().split("\t")[0]
                item2id[item] = idx
                id2item.append(item)
        return item2id, id2item

    def read_triplets(self, path):
        triplets = []
        print(f"Reading {path}...")
        with open(path, "r") as f:
            for line in tqdm(f.readlines()):
                h,r,t = line.strip().split()
                triplets.append((int(h), int(r), int(t)))
        return triplets

    def check_dups(self):
        train_set = set(self.train_pos + self.train_neg)
        valid_set = set(self.valid_pos + self.valid_neg)
        test_set = set(self.test_pos + self.test_neg)
        for triplet in valid_set:
            assert triplet not in train_set, triplet
        for triplet in test_set:
            assert triplet not in train_set, triplet
            assert triplet not in valid_set, triplet