from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine import engine
from torchreid.losses import CrossEntropyLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageSoftmaxEngine(engine.Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_cpu (bool, optional): use cpu. Default is False.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torch
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(self, datamanager, model, optimizer, scheduler=None, use_cpu=False,
                 label_smooth=True, experiment=None):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_cpu, experiment)
        
        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses = AverageMeter()
        rank_1 = AverageMeter()
        rank_2 = AverageMeter()
        rank_3 = AverageMeter()
        rank_4 = AverageMeter()
        rank_5 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch+1)<=fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)
            num_batches = len(trainloader)
            global_step = num_batches * epoch + batch_idx

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
             
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self._compute_loss(self.criterion, outputs, pids)
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), pids.size(0))
            accs = metrics.accuracy(outputs, pids)
            rank_1.update(accs[0].item())
            rank_2.update(accs[1].item())
            rank_3.update(accs[2].item())
            rank_4.update(accs[3].item())
            rank_5.update(accs[4].item())

            # write to Tensorboard & comet.ml
            for i,r in enumerate(accs):
                r = float(r)
                self.writer.add_scalar('ranks/rank-'+str(i+1),r,global_step)
                self.experiment.log_metric('ranks/rank-'+str(i+1),r,step=global_step)
            
            # self.writer.add_scalar('ranks-avg/rank-1',rank_1.avg,global_step)
            # self.writer.add_scalar('ranks-avg/rank-2',rank_2.avg,global_step)
            # self.writer.add_scalar('ranks-avg/rank-3',rank_3.avg,global_step)
            # self.writer.add_scalar('ranks-avg/rank-4',rank_4.avg,global_step)
            # self.writer.add_scalar('ranks-avg/rank-5',rank_5.avg,global_step)
                
            self.writer.add_scalar('optim/loss',losses.val,global_step) # loss, loss.item() or losses.val ??
            # self.writer.add_scalar('optim/loss-avg',losses.avg,global_step)
            self.experiment.log_metric('optim/loss',losses.val,step=global_step)

            self.writer.add_scalar('optim/lr',self.optimizer.param_groups[0]['lr'],global_step)
            self.experiment.log_metric('optim/lr',self.optimizer.param_groups[0]['lr'],step=global_step)
            


            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                num_batches = len(trainloader)
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Rank-1 {r1.val:.2f} ({r1.avg:.2f})\t'
                      'Rank-2 {r2.val:.2f} ({r2.avg:.2f})\t'
                      'Rank-3 {r3.val:.2f} ({r3.avg:.2f})\t'
                      'Rank-4 {r4.val:.2f} ({r4.avg:.2f})\t'
                      'Rank-5 {r5.val:.2f} ({r5.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'Eta {eta}'.format(
                      epoch+1, max_epoch, batch_idx+1, len(trainloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      r1=rank_1,
                      r2=rank_2,
                      r3=rank_3,
                      r4=rank_4,
                      r5=rank_5,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str
                    )
                )
                self.writer.add_scalar('eta',eta_seconds,global_step)
                self.experiment.log_metric('eta',eta_seconds,step=global_step)
            
            end = time.time()

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(losses.val)
        elif self.scheduler is not None:
            self.scheduler.step()
        