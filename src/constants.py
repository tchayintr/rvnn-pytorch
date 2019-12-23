import progressbar

global __version
__version__ = '0.1.0'


# for analyser

LOG_DIR = 'log'
MODEL_DIR = 'models/main'


# for token

UNK_TOKEN = 'UNK'


# misc
PROGRESSBAR_WIDGETS = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
