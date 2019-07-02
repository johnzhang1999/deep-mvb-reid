import sys
import os
import os.path as osp
import warnings
import time

from comet_ml import Experiment
import torch
import torch.nn as nn

from default_parser import (
    init_parser, imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs
)
import torchreid
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info,
    change_lr_to
)

parser = init_parser()
args = parser.parse_args()

# comet.ml experiment logging

if args.no_comet:
    experiment = Experiment(api_key="jYkp7GiEE17RfR1iGGvF2rMTB",
                        project_name="mvbchallenge", workspace="johnzhang1999",
                        parse_args=False, disabled=True)
else:
    experiment = Experiment(api_key="jYkp7GiEE17RfR1iGGvF2rMTB",
                        project_name="mvbchallenge", workspace="johnzhang1999",
                        parse_args=False)

name = args.arch + '_' + args.sources[0] + '_' + str(args.lr) + '_' + str(args.batch_size)
experiment.set_name(name)
if args.resume:
    experiment.add_tag('resume')
experiment.log_parameters(args.__dict__)
# asset logging is BUG-gy!
experiment.log_asset_folder(osp.expanduser('log/'))
experiment.log_asset_folder(osp.expanduser('runs/'))


def build_datamanager(args):
    if args.app == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(args))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(args))


def build_engine(args, datamanager, model, optimizer, scheduler, experiment=experiment):
    if args.app == 'image':
        if args.loss == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_cpu=args.use_cpu,
                label_smooth=args.label_smooth,
                experiment=experiment
            )
        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=args.margin,
                weight_t=args.weight_t,
                weight_x=args.weight_x,
                scheduler=scheduler,
                use_cpu=args.use_cpu,
                label_smooth=args.label_smooth,
                experiment=experiment
            )
    
    else:
        if args.loss == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_cpu=args.use_cpu,
                label_smooth=args.label_smooth,
                pooling_method=args.pooling_method
            )
        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=args.margin,
                weight_t=args.weight_t,
                weight_x=args.weight_x,
                scheduler=scheduler,
                use_cpu=args.use_cpu,
                label_smooth=args.label_smooth
            )

    return engine


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available() and not args.use_cpu
    log_name = 'test.log' if args.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('** Arguments **')
    arg_keys = list(args.__dict__.keys())
    arg_keys.sort()
    for key in arg_keys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('\n')
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    datamanager = build_datamanager(args)
    
    print('Building model: {}'.format(args.arch))
    model = torchreid.models.build_model(
        name=args.arch,
        num_classes=datamanager.num_train_pids,
        loss=args.loss.lower(),
        pretrained=(not args.no_pretrained),
        use_gpu=use_gpu
    )
    num_params, flops = compute_model_complexity(model, (1, 3, args.height, args.width))
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)
    
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(args))

    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, scheduler=scheduler, optimizer=optimizer)
    
    # lr changing is BUG-gy!
    if args.lr != optimizer.param_groups[0]['lr']:
        old_lr = optimizer.param_groups[0]['lr']
        change_lr_to(optimizer,args.lr)
        print('Changed optimzer lr from {} to {}.'.format(old_lr,args.lr))

    print('Building {}-engine for {}-reid'.format(args.loss, args.app))
    engine = build_engine(args, datamanager, model, optimizer, scheduler, experiment)

    engine.run(**engine_run_kwargs(args))


if __name__ == '__main__':
    main()