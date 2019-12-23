from datetime import datetime
import progressbar
import random
import sys

import torch
from torch.nn.utils import clip_grad_norm_

import constants
from recursive_nn import RecursiveNN
from sentiment_tree import SentimentTree



class Trainer(object):
    def __init__(self, args, logger=sys.stderr):
        self.args = args
        self.start_time = datetime.now().strftime('%Y%m%d_%H%M')
        self.logger = logger    # output execute log
        self.reporter = None    # output evaluation results
        self.train = None
        self.valid = None
        self.test = None
        self.hparams = None
        self.model = None
        self.widgets = None
        self.optimizer = None
        self.best_all = None
        self.best_root = None
        self.current_epoch = None


        self.log('Start time: {}\n'.format(self.start_time))
        if not self.args.quiet:
            self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), mode='a')


    def report(self, message):
        if not self.args.quiet:
            print(message, file=self.reporter)


    def log(self, message=''):
        print(message, file=self.logger)


    def close(self):
        if not self.args.quiet:
            self.reporter.close()


    def init_hyperparameters(self):
        self.hparams = {
            'epoch': self.args.epoch,
            'embed_dim': self.args.embed_dim,
            'num_class': self.args.num_class,
            'learning_rate': self.args.learning_rate,
            'momentum': self.args.momentum,
            'max_norm': self.args.max_norm,
            'norm_type': self.args.norm_type,
        }

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def load_training_and_validation_data(self):
        self.load_data('train')
        self.load_data('valid')


    def load_test_data(self):
        self.load_data('test')


    def load_data(self, data_type):
        if data_type == 'train':
            data_path = self.args.train_data
            data = SentimentTree.load_trees(self.args.train_data, vocab_output=self.args.vocab_indices)
            self.train = data

        elif data_type == 'valid':
            data_path = self.args.valid_data
            data = SentimentTree.load_trees(self.args.valid_data, vocab_indices=self.args.vocab_indices)
            self.valid = data

        elif data_type == 'test':
            data_path = self.args.test_data
            data = None
            self.test = data

        else:
            print('Error: incorrect data type: {}'.format(), file=sys.stderr)
            sys.exit()

        self.log('Load {} data: {}'.format(data_type, data_path))


    def setup_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams['learning_rate'], momentum=self.hparams['momentum'])


    def run(self):
        if self.args.cuda:
            self.log('# Running Model with CUDA')
            self.model = RecursiveNN(SentimentTree.vocab_size, embed_dim=self.args.embed_dim, num_class=self.args.num_class, use_cuda=True).cuda()
        else:
            self.log('# Running Model without CUDA')
            self.model = RecursiveNN(SentimentTree.vocab_size, embed_dim=self.args.embed_dim, num_class=self.args.num_class, use_cuda=False)

        self.setup_optimizer()
        self.log('# Model: {}'.format(self.model))
        self.log('# Optimizer: {}'.format(self.optimizer))

        max_epoch = self.hparams['epoch']
        best_all = 0.0
        best_root = 0.0
        loss = 0.0

        for epoch in range(max_epoch):
            print("\n# Epoch {}".format(epoch))
            pbar = progressbar.ProgressBar(widgets=constants.PROGRESSBAR_WIDGETS, maxval=len(self.train)).start()

            for step, tree in enumerate(self.train):
                predictions, loss = self.model.get_loss(tree)
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.hparams['max_norm'], norm_type=self.hparams['norm_type'])
                self.optimizer.step()
                pbar.update(step)
            pbar.finish()

            correct_root, correct_all = self.model.evaluate(self.valid)

            if best_all < correct_all:
                best_all = correct_all
            if best_root < correct_root:
                best_root = correct_root

            self.best_all = best_all
            self.best_root = best_root
            self.current_epoch = epoch

            self.log('# loss: {}'.format(loss))
            self.log('# validation: all-node accuracy: {}, (best: {})'.format(str(round(correct_all, 2)), str(round(self.best_all, 2))))
            self.log('# validation: root accuracy: {}, (best: {})'.format(str(round(correct_root, 2)), str(round(self.best_root, 2))))

            random.shuffle(self.train)


    def summary(self):
        best_all = str(round(self.best_all, 2))
        best_root = str(round(self.best_root, 2))

        self.log('# Summary')
        self.log('All-node accuracy: {}'.format(best_all))
        self.log('Root accuracy: {}'.format(best_root))
        self.log('Run {} Epoch(s)'.format(self.current_epoch + 1))

        self.report('[SUMMARY] Summary')
        self.report('[SUMMARY] All-node accuracy: {}'.format(best_all))
        self.report('[SUMMARY] Root accuracy: {}'.format(best_root))
        self.report('[SUMMARY] Run {} Epoch(s)'.format(self.current_epoch + 1))
