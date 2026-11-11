import socket
from diffuser.utils import watch

#------------------------ base ------------------------#

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 128,
        'n_diffusion_steps': 50,
        'action_weight': 10,
        'loss_weights': {
            0: 6.0,
            1: 6.0,
            2: 2.0,
            3: 2.0,
            4: 2.0,
            5: 2.0,
            **{i: 0.15 for i in range(11, 59)},
        },
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim': 64,
        'dim_mults': (1, 2, 4),
        'attention': True,
        'renderer': 'utils.NavigationRenderer',

        ## dataset
        'loader': 'datasets.NavigationGoalDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 328,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/navigation',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'l2',
        'n_train_steps': 15000,
        'batch_size': 64,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 3000,
        'sample_freq': 3000,
        'n_saves': 10,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },
}
