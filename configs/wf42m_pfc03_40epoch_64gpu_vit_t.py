from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_t_dp005_mask0"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 384
config.optimizer = "adamw"
config.lr = 0.001
config.verbose = 2000
config.dali = False

'''config.rec = "/train_tmp/WebFace42M"
config.num_classes = 2059906
config.num_image = 42474557
config.num_epoch = 40
config.warmup_epoch = config.num_epoch // 10
config.val_targets = []'''


config.rec= "/store01/flynn/darun/uerc2023/data/public/"
config.rec_val= "/afs/crc.nd.edu/user/d/darun/insightface/recognition/arcface_torch/kagan/deeksha/"
config.num_classes = 1310
config.num_image = 248655
config.num_epoch = 20
config.warmup_epoch = config.num_epoch // 10
config.val_targets = ['awe_pairs_176_96']
config.dali_aug = False

config.output= f'results/{config.network}_exp_0'

