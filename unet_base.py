import argparse
import random
import numpy as np
import tensorflow as tf
import logging
from unet.model import BoxUnet
from unet.data import data_fn
import os
import json
import time

from kibernetika.kibernetika_utils import klclient


def null_dataset():
    def _input_fn():
        return None
    return _input_fn

def export(checkpoint_dir, params):
    if os.environ.get('TRAINING_DIR', '') != '' and os.environ.get('BASE_TASK_BUILD_ID', '') != '':
        checkpoint_dir = os.environ['TRAINING_DIR'] + '/' + os.environ['BASE_TASK_BUILD_ID']
    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
    )
    params['batch_size']=1
    feature_placeholders = {
        'image': tf.placeholder(tf.float32, [1, None, None, 3], name='image'),
    }
    receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders)
    net = BoxUnet(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )
    export_path = net.export_savedmodel(
        checkpoint_dir,
        receiver,
    )
    export_path = export_path.decode("utf-8")
    klclient.update_task_info({'model_path': export_path})


def train(mode, checkpoint_dir, params):
    logging.info("start build  model")

    save_summary_steps = params['save_summary_steps']
    save_checkpoints_secs = params['save_checkpoints_secs'] if params['save_checkpoints_steps'] is None else None
    save_checkpoints_steps = params['save_checkpoints_steps']

    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=params['keep_checkpoint_max'],
        log_step_count_steps=params['log_step_count_steps'],
    )
    fn = data_fn(params, mode == 'train')
    net = BoxUnet(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
        warm_start_from=params['warm_start_from']

    )
    logging.info("Start %s mode type %s", mode, conf.task_type)
    if mode == 'train' and conf.task_type != 'ps':
        if conf.master != '':
            train_fn = fn
            train_spec = tf.estimator.TrainSpec(input_fn=train_fn)
            eval_fn = null_dataset()
            eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn, steps=1, start_delay_secs=10, throttle_secs=10)
            tf.estimator.train_and_evaluate(net, train_spec, eval_spec)
        else:
            net.train(input_fn=fn)
    else:
        train_fn = null_dataset()
        train_spec = tf.estimator.TrainSpec(input_fn=train_fn)
        eval_fn = fn
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn, steps=1, start_delay_secs=10, throttle_secs=10)
        tf.estimator.train_and_evaluate(net, train_spec, eval_spec)


def main(args):
    params = {
        'num_pools': args.num_pools,
        'num_epochs': args.num_epochs,
        'drop_prob': args.drop_prob,
        'num_chans': args.num_chans,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'weight_decay': args.weight_decay,
        'checkpoint': str(args.checkpoint_dir),
        'resolution': args.resolution,
        'save_summary_steps': args.save_summary_steps,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'keep_checkpoint_max': args.keep_checkpoint_max,
        'log_step_count_steps': args.log_step_count_steps,
        'warm_start_from': args.warm_start_from,
        'use_seed': False,
        'resolution': args.resolution,
        'data_set': args.data_set,
        'limit': args.limit,
        'optimizer': args.optimizer,
    }
    if args.export:
        export(args.checkpoint_dir, params)
        return
    if not tf.gfile.Exists(args.checkpoint_dir):
        tf.gfile.MakeDirs(args.checkpoint_dir)
    if args.worker:
        klclient.update_task_info({
            'num-pools': args.num_pools,
            'drop-prob': args.drop_prob,
            'num-chans': args.num_chans,
            'batch-size': args.batch_size,
            'lr.lr': args.lr,
            'lr.lr-step-size': args.lr_step_size,
            'lr.lr-gamma': args.lr_gamma,
            'weight-decay': args.weight_decay,
            'checkpoint_path': str(args.checkpoint_dir),
            'resolution': args.resolution
        })
        train('train', args.checkpoint_dir, params)
    else:
        cluster = {'chief': ['fake_worker1:2222'],
                   'ps': ['fake_ps:2222'],
                   'worker': ['fake_worker2:2222']}
        os.environ['TF_CONFIG'] = json.dumps(
            {
                'cluster': cluster,
                'task': {'type': 'evaluator', 'index': 0}
            })
        train('eval', args.checkpoint_dir, params)


def create_arg_parser():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs')
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')

    parser.add_argument('--data_set', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--limit', default=-1, type=int, help='Limit train')
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=10,
        help="Log summary every 'save_summary_steps' steps",
    )
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=600,
        help="Save checkpoints every 'save_checkpoints_secs' secs.",
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=100,
        help="Save checkpoints every 'save_checkpoints_steps' steps",
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=2,
        help='The maximum number of recent checkpoint files to keep.',
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=1,
        help='The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.',
    )
    parser.add_argument(
        '--warm_start_from',
        type=str,
        default=None,
        help='Warm start from',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='RMSPropOptimizer',
        help='Optimizer',
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(worker=False)
    group.set_defaults(evaluator=False)
    group.set_defaults(test=False)
    group.add_argument('--worker', dest='worker', action='store_true',
                       help='Start in Worker(training) mode.')
    group.add_argument('--evaluator', dest='evaluator', action='store_true',
                       help='Start in evaluation mode')
    group.add_argument('--test', dest='test', action='store_true',
                       help='Test mode')
    group.add_argument('--export', dest='export', action='store_true',
                       help='Export model')
    return parser


if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    args = create_arg_parser().parse_args()
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))
    main(args)
