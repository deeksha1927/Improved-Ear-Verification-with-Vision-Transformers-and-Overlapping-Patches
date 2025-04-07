from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_h"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.5
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128
config.optimizer = "adamw"
config.lr = 0.001
config.verbose = 2000
config.dali = False

config.rec= "/store01/flynn/darun/awe_lr_rotated_cropped/"
config.num_classes = 200
config.num_image = 1000
config.num_epoch = 100
config.warmup_epoch = config.num_epoch // 10
config.val_targets = ['earvn_pairs_112_112']
config.dali_aug = False

config.output= f'results/{config.network}_200_1000_100_112x112'
