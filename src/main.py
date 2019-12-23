from arguments import ArgumentLoader
from core import Core
from trainer import Trainer



class Main(Core):
    def __init__(self):
        super().__init__()


    def get_args(self):
        parser = ArgumentLoader()
        args = parser.parse_args()
        return args


    def get_trainer(self, args):
        trainer = Trainer(args)
        return trainer



if __name__ == '__main__':
    analyzer = Main()
    analyzer.run()
