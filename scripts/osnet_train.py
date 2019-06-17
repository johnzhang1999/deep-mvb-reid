import torchreid
datamanager = torchreid.data.ImageDataManager(
    root='/root/mvb/data',
    sources='market1501',
    height=256,
    width=256,
    batch_size=110,
    market1501_500k=False,
    random_erase=True,
    workers=16
)

model = torchreid.models.build_model(
    name='osnet',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=False
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.1
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[1,5,80,150,225,300]
)

start_epoch = torchreid.utils.resume_from_checkpoint(
    '/root/mvb/deep-person-reid/scripts/log/osnet-mvb-softmax-run-2/model.pth.tar-10',
    model,
    optimizer
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir='log/osnet-mvb-softmax-run-2',
    ranks=[1,2,3,5,10,20],
    max_epoch=350,
    eval_freq=10,
    start_eval=60,
    print_freq=10,
    test_only=False,
    start_epoch=start_epoch
)