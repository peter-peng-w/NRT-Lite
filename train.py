import os
import argparse

import config
from main import init, load_dataset, get_n_ui
from trainer import MultiTaskTrainer, RankerTrainer, ReviewTrainer, NRTTaskTrainer

DIR_PATH = os.path.dirname(__file__)

KARGS_LOG_KEYS = {'batch_size', 'lr', 'l2', 'clip', 'rank_loss_type', 'loss_lambda', 'max_iters', 'max_length', 'grp_config', 'n_mc_rollout'}


def config_to_kargs(model_config):
    return dict(
        batch_size=model_config.BATCH_SIZE,
        lr=model_config.LR,
        l2=model_config.L2_PENALTY,
        clip=model_config.CLIP,
        patience=config.PATIENCE,
        max_iters=model_config.MAX_ITERS,
        save_every=config.SAVE_EVERY,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model name to save/load checkpoints')
    parser.add_argument('-c', '--checkpoint')

    args = parser.parse_args()

    model, misc = init(args.model, args.checkpoint)
    voc, checkpoint, ckpt_mng, model_config = (
        misc[k] for k in ['voc', 'checkpoint', 'ckpt_mng', 'model_config']
    )

    trainer_args = (model, ckpt_mng)

    kargs = config_to_kargs(model_config)

    if model_config.TRAINING_TASK == 'rate':
        Trainer = RankerTrainer
        kargs['rank_loss_type'] = model_config.RANK_LOSS_TYPE
        kargs['loss_lambda'] = model_config.LOSS_LAMBDA
    elif model_config.TRAINING_TASK == 'review':
        Trainer = ReviewTrainer
        kargs['voc'] = voc
    elif model_config.TRAINING_TASK == 'nrt':
        Trainer = NRTTaskTrainer
        kargs['loss_lambda'] = model_config.LOSS_LAMBDA
    else:
        Trainer = MultiTaskTrainer
        kargs['voc'] = voc
        kargs['loss_lambda'] = model_config.LOSS_LAMBDA
        kargs['rank_loss_type'] = model_config.RANK_LOSS_TYPE

    if 'rank_loss_type' in kargs and kargs['rank_loss_type']:
        kargs['grp_config'] = config.LOSS_TYPE_GRP_CONFIG[kargs['rank_loss_type']]

    print(f'Training config:', {k: v for k, v in kargs.items() if k in KARGS_LOG_KEYS})

    trainer = Trainer(
        *trainer_args,
        **kargs
    )

    if checkpoint:
        trainer.resume(checkpoint)

    train_dataset = load_dataset('train')
    dev_dataset = load_dataset('dev')

    # Ensure dropout layers are in train mode
    model.train()

    trainer.train(train_dataset, dev_dataset)


if __name__ == '__main__':
    main()
