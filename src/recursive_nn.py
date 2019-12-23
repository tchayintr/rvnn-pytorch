import progressbar
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import constants



class RecursiveNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, use_cuda):
        super(RecursiveNN, self).__init__()

        self.embedding = nn.Embedding(int(vocab_size), embed_dim)
        self.W = nn.Linear(2 * embed_dim, embed_dim, bias=True)
        self.projection = nn.Linear(embed_dim, num_class, bias=True)
        self.activation = F.relu
        self.node_probabilities = []
        self.labels = []
        self.use_cuda = use_cuda


    def traverse(self, node):
        if node.is_leaf():
            current_node = self.activation(self.embedding(self.var(torch.LongTensor([node.get_leaf_word()]))))
        else:
            current_node = self.activation(self.W(torch.cat((self.traverse(node.left()), self.traverse(node.right())), 1)))

        self.node_probabilities.append(self.projection(current_node))
        self.labels.append(torch.LongTensor([node.label()]))
        return current_node


    def forward(self, x):
        self.node_probabilities = []
        self.labels = []
        self.traverse(x)
        self.labels = self.var(torch.cat(self.labels))
        return torch.cat(self.node_probabilities)


    def get_loss(self, tree):
        nodes = self.forward(tree)
        predictions = nodes.max(dim=1)[1]
        loss = F.cross_entropy(input=nodes, target=self.labels)
        return predictions, loss


    def evaluate(self, trees):
        pbar = progressbar.ProgressBar(widgets=constants.PROGRESSBAR_WIDGETS, maxval=len(trees)).start()
        n = n_all = correct_root = correct_all = 0.0

        for j, tree in enumerate(trees):
            predictions, loss = self.get_loss(tree)
            correct = (predictions.data == self.labels.data)
            correct_all += correct.sum()
            n_all += correct.squeeze().size()[0]
            correct_root += float(correct.squeeze()[-1])
            n += 1
            pbar.update(j)
        pbar.finish()

        correct_root = float(correct_root) / n
        correct_all = float(correct_all) / n_all

        return correct_root, correct_all


    def var(self, v):
        if self.use_cuda:
            return Variable(v.cuda())
        else:
            return Variable(v)
