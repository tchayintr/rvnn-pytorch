from nltk.tree import ParentedTree
import pickle

import constants



class SentimentTree(ParentedTree):
    def __init__(self, node, children=None):
        super(SentimentTree, self).__init__(node, children)


    def left(self):
        return self[0]


    def right(self):
        return self[1]


    def is_leaf(self):
        return self.height() == 2


    def get_leaf_word(self):
        return self[0]


    @staticmethod
    def load_trees(path, vocab_output=None, vocab_indices=None):
        if vocab_indices is None:
            return SentimentTree.construct_vocab_and_get_trees(path, vocab_output=vocab_output)
        else:
            return SentimentTree.get_trees_given_vocab(path, vocab_indices)


    @staticmethod
    def get_trees_given_vocab(path, vocab_indices):
        trees = []
        vocab_indices = pickle.load(open(vocab_indices, 'rb'))

        with open(path, 'rt') as f:
            for line in f:
                tree = SentimentTree.fromstring(line)
                SentimentTree.map_tree_nodes(tree, vocab_indices)
                SentimentTree.cast_labels_to_int(tree)
                trees.append(tree)

        SentimentTree.vocab_size = len(vocab_indices)

        return trees


    @staticmethod
    def construct_vocab_and_get_trees(path, vocab_output=None):
        trees = []
        vocab = set()

        with open(path, "rt") as f:
            for line in f:
                tree = SentimentTree.fromstring(line)
                trees.append(tree)
                vocab.update(tree.leaves())

        vocab_indices = dict(zip(vocab, range(len(vocab))))
        vocab_indices[constants.UNK_TOKEN] = len(vocab)

        if vocab_output is not None:
            with open(vocab_output, 'wb') as fp:
                pickle.dump(vocab_indices, fp)

        for tree in trees:
            SentimentTree.map_tree_nodes(tree, vocab_indices)
            SentimentTree.cast_labels_to_int(tree)
        SentimentTree.vocab_size = len(vocab_indices)

        return trees


    @staticmethod
    def map_tree_nodes(tree, vocab_indices):
        for leaf_pos in tree.treepositions('leaves'):
            if tree[leaf_pos] in vocab_indices:
                tree[leaf_pos] = vocab_indices[tree[leaf_pos]]
            else:
                tree[leaf_pos] = vocab_indices[constants.UNK_TOKEN]


    @staticmethod
    def cast_labels_to_int(tree):
        for subtree in tree.subtrees():
            subtree.set_label(int(subtree.label()))
