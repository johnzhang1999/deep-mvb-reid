from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import warnings
import csv

from torchreid.data.datasets import ImageDataset


class Market1501(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'MVB_train'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)
    
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')
        
        self.train_dir = osp.join(self.data_dir, 'Info/sa_train.csv')
        self.query_dir = osp.join(self.data_dir, 'Info/sa_query.csv')
        self.gallery_dir = osp.join(self.data_dir, 'Info/sa_gallery.csv')
        # self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ] 
        # if self.market1501_500k:
        #     required_files.append(self.extra_gallery_dir)
        # self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        ret = []
        real_ret = []
        pid_container = set()
        with open(osp.join(dir_path)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                filename,pid,cam = row[0],row[1],row[2]
                pid_container.add(pid)
                ret.append(('/root/mvb/data/MVB_train/Image/'+filename,pid,int(cam)))
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        # print(pid2label)
        if relabel:
            for path,pid,cam in ret:
                real_ret.append((path,pid2label[pid],cam))
        else:
            for path,pid,cam in ret:
                real_ret.append((path,int(pid),cam))
        return real_ret