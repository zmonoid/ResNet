import argparse, logging, os
import mxnet as mx
import importlib

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def multi_factor_scheduler(begin_epoch,
                           epoch_size,
                           step=[60, 75, 90],
                           factor=0.1):
    step_ = [
        epoch_size * (x - begin_epoch) for x in step if x - begin_epoch > 0
    ]
    return mx.lr_scheduler.MultiFactorScheduler(
        step=step_, factor=factor) if len(step_) else None


def main(kv):
    if args.network.startswith('resnet'):
        if args.aug_level == 1:
            img_shape = (3, 256, 256)
        else:
            img_shape = (3, 480, 480)
        crop_shape = (3, 224, 224)

        name = args.network.split('-')[0]
        depth = args.network.split('-')[1]
        symbol = importlib.import_module('symbols.' + name).get_symbol(
            args.num_classes, int(depth), str(crop_shape)[1:-1])
        train_rec = 'train_%d.rec' % img_shape[1]
        val_rec = 'val_256.rec'

    elif args.network.startswith('inception'):
        img_shape = (3, 337, 337)
        crop_shape = (3, 299, 299)
        symbol = importlib.import_module('symbols.' + args.network).get_symbol(
            args.num_classes)
        train_rec = 'train_%d.rec' % img_shape[1]
        val_rec = 'val_337.rec'
    elif args.network.startswith('vgg'):
        img_shape = (3, 256, 256)
        crop_shape = (3, 224, 224)
        symbol = importlib.import_module('symbols.' + args.network).get_symbol(
            args.num_classes)
        train_rec = 'train_%d.rec' % img_shape[1]
        val_rec = 'val_256.rec'

    if args.gpus is None:
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = max(
        int(args.num_examples / args.batch_size / kv.num_workers), 1)

    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
    save_model_prefix = args.save_model_prefix
    # save_model_prefix = "{}-{}-{}-{}".format(
    #     args.save_model_prefix, data_type, args.depth, kv.rank)

    checkpoint = mx.callback.do_checkpoint(save_model_prefix)
    arg_params = None
    aux_params = None
    if args.load_model_prefix is not None:
        bla, arg_params, aux_params = mx.model.load_checkpoint(
            args.load_model_prefix, args.model_load_epoch)
        print bla

    train = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(args.data_dir, train_rec),
        label_width=1,
        data_name='data',
        label_name='softmax_label',
        data_shape=crop_shape,
        batch_size=args.batch_size,
        pad=0,
        fill_value=127,  # only used when pad is valid
        rand_crop=True,
        max_random_scale=1.0,
        min_random_scale=1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio=0 if args.aug_level == 1 else 0.25,
        random_h=0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s=0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l=0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle=0 if args.aug_level <= 2 else 10,
        max_shear_ratio=0 if args.aug_level <= 2 else 0.1,
        rand_mirror=True,
        shuffle=True,
        num_parts=kv.num_workers,
        part_index=kv.rank)
    val = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(args.data_dir, val_rec),
        label_width=1,
        data_name='data',
        label_name='softmax_label',
        batch_size=args.batch_size,
        data_shape=crop_shape,
        rand_crop=False,
        rand_mirror=False,
        num_parts=kv.num_workers,
        part_index=kv.rank)
    model = mx.model.FeedForward(
        ctx=devs,
        symbol=symbol,
        arg_params=arg_params,
        aux_params=aux_params,
        num_epoch=120,
        begin_epoch=begin_epoch,
        learning_rate=args.lr,
        momentum=args.mom,
        wd=args.wd,
        optimizer='nag',
        # optimizer          = 'sgd',
        initializer=mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2),
        lr_scheduler=multi_factor_scheduler(
            begin_epoch, epoch_size, step=[30, 60, 90], factor=0.1), )
    model.fit(
        X=train,
        eval_data=val,
        eval_metric=['acc', mx.metric.create('top_k_accuracy', top_k=5)],
        kvstore=kv,
        batch_end_callback=mx.callback.Speedometer(args.batch_size,
                                                   args.frequent),
        epoch_end_callback=checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command for training resnet-v2")
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/',
        help='the input data directory')
    parser.add_argument(
        '--list-dir',
        type=str,
        default='./data/',
        help='the directory which contain the training list file')
    parser.add_argument(
        '--lr', type=float, default=0.1, help='initialization learning reate')
    parser.add_argument(
        '--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument(
        '--bn-mom',
        type=float,
        default=0.9,
        help='momentum for batch normlization')
    parser.add_argument(
        '--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='the batch size')
    parser.add_argument(
        '--workspace',
        type=int,
        default=512,
        help='memory space size(MB) used in convolution, if xpu '
        ' memory is oom, then you can try smaller vale, such as --workspace 256'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='inception-bn',
        choices=[
            'alexnet', 'googlenet', 'inception-bn', 'inception-v3',
            'resnet-50', 'resnet-101', 'vgg'
        ],
        help='the cnn to use')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=1000,
        help='the class number of your task')
    parser.add_argument(
        '--aug-level',
        type=int,
        default=2,
        choices=[1, 2, 3],
        help='level 1: use only random crop and random mirror\n'
        'level 2: add scale/aspect/hsv augmentation based on level 1\n'
        'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument(
        '--num-examples',
        type=int,
        default=1281167,
        help='the number of training examples')
    parser.add_argument(
        '--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument(
        '--load-model-prefix',
        type=str,
        default=None,
        help='the prefix of the model to load')
    parser.add_argument(
        '--save-model-prefix',
        type=str,
        help='the prefix of the model to save')
    parser.add_argument(
        '--model-load-epoch',
        type=int,
        default=0,
        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument(
        '--save-log-prefix', type=str, help='the name of log file')
    parser.add_argument(
        '--frequent', type=int, default=50, help='frequency of logging')
    args = parser.parse_args()

    kv = mx.kvstore.create(args.kv_store)
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if args.save_log_prefix is not None:
        log_file_full_name = args.save_log_prefix + '.log'
        logger = logging.getLogger()

        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)
        logging.info(args)
    main(kv)
