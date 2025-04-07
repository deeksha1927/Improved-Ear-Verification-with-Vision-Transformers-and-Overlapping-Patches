from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_b_dp005_mask_005"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128
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

config.rec= "/store01/flynn/darun/rotated_Ears/rotated-ears/uerc2023_warped_lr/"
config.num_classes = 2403
config.num_image = 247654
config.num_epoch = 2
config.warmup_epoch = config.num_epoch // 10
config.val_targets = ['earvn_warped_lr_112_112']
config.dali_aug = False
config.rec_val="/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/kagan/deeksha/"

#config.output= f'results/{config.network}_uerc-awe-rotated-cropped_2404_247655_100_112x112_p{config.patch_size}_s{config.stride}'
