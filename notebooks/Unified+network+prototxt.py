import os
import sys
import pprint

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
add_path('/home/ubuntu/src/py-faster-rcnn/caffe-fast-rcnn/python')
add_path('/home/ubuntu/src/py-faster-rcnn/lib')
import caffe
from datasets.factory import get_imdb
from fast_rcnn.test import test_net
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
print "Loaded caffe version {:s} from {:s}.".format(caffe.__version__, caffe.__path__[0])

# configure plotting
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()

this_dir = os.getcwd()
data_dir = os.path.join(this_dir, '..', 'data', 'voc')
model_dir = os.path.join(this_dir, '..', 'models', 'ZF')

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

solver_file = os.path.join(model_dir, 'solver.prototxt')

cfg_from_file('../experiments/cfgs/faster_rcnn_end2end.yml')
print('Using config:')
pprint.pprint(cfg)
np.random.seed(cfg.RNG_SEED)
caffe.set_random_seed(cfg.RNG_SEED)
imdb, roidb = combined_roidb('voc_2007_train')
output_dir = get_output_dir(imdb)
print output_dir

pretrained_caffemodel = '/home/ubuntu/src/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel'
train_net(solver_file, roidb, output_dir,                                                                                        
          pretrained_model=pretrained_caffemodel,                                                                                
          max_iters=100000)     


