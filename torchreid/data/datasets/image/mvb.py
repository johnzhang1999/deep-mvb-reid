from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import warnings
import csv

from torchreid.data.datasets import ImageDataset


class MVB(ImageDataset):

    dataset_dir = 'MVB_train'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
    
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('MVB dataset load error.')
        
        self.train_dir = osp.join(self.data_dir, 'Info/sa_train.csv')
        self.query_dir = osp.join(self.data_dir, 'Info/sa_query.csv')
        self.gallery_dir = osp.join(self.data_dir, 'Info/sa_gallery.csv')

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(MVB, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        ret = []
        real_ret = []
        pid2label = {}
        idx = 0
        with open(osp.join(dir_path)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                filename,pid,cam = row[0],row[1],row[2]
                if pid not in pid2label:
                	pid2label[pid] = idx
                	idx += 1
                abs_path = osp.join(self.dataset_dir,'Image/',filename)
                ret.append((abs_path,pid,int(cam)))
        if relabel:
            for path,pid,cam in ret:
                real_ret.append((path,pid2label[pid],cam))
        else:
            for path,pid,cam in ret:
                real_ret.append((path,int(pid),cam))
        return real_ret