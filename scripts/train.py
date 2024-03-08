import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

import argparse
import utils
from GSGM import GSGM
from deepsets_cond import DeepSetsAttClass
from tensorflow.keras.callbacks import ModelCheckpoint
import horovod.tensorflow.keras as hvd

#tf.random.set_seed(1233)
if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()    
    parser.add_argument('--folder', default='/pscratch/sd/v/vmikuni/TOP/', help='Path containing the training files')
    parser.add_argument('--dataset', default='gluon', help='Which dataset to train')
    parser.add_argument("--batch", type=int, default=250, help="Batch size")
    parser.add_argument("--epoch", type=int, default=150, help="Max epoch")
    parser.add_argument("--warm_epoch", type=int, default=10, help="Warm up epochs")
    parser.add_argument("--stop_epoch", type=int, default=30, help="Epochs before reducing lr")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")


    flags = parser.parse_args()
    
    train = utils.DataLoader(os.path.join(flags.folder, '{}_train.h5'.format(flags.dataset)),
                             flags.batch,hvd.rank(),hvd.size())
    test = utils.DataLoader(os.path.join(flags.folder, '{}_test.h5'.format(flags.dataset)),
                            flags.batch,hvd.rank(),hvd.size())    
    
    scale_lr = flags.lr*np.sqrt(hvd.size())


    model = GSGM(num_feat = train.num_feat,
                 num_jet = train.num_jet,
                 num_part=train.num_part)

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=flags.lr,
        warmup_target = scale_lr,
        warmup_steps= flags.warm_epoch*train.nevts//flags.batch//hvd.size(),
        decay_steps=flags.epoch*train.nevts//flags.batch//hvd.size(),
        #alpha = 1e-3,
    )

    opt = keras.optimizers.Lion(
        learning_rate = lr_schedule,
        weight_decay=1e-4,
        beta_1=0.95,
    )    
    opt = hvd.DistributedOptimizer(opt)
    
    model.compile(weighted_metrics=[],
                  optimizer=opt,
                  #run_eagerly=True,
                  experimental_run_tf_function=False
                  )
    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        EarlyStopping(patience=flags.stop_epoch,restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss',patience=200, min_lr=1e-6), #The cosine schedule already controls the LR, mostly used to print the LR value during training
    ]

        
    if hvd.rank()==0:
        checkpoint_folder = '../checkpoints/{}.weights.h5'.format(flags.dataset)
        checkpoint = ModelCheckpoint(checkpoint_folder,mode='auto',
                                     save_best_only=True,period=1,
                                     save_weights_only=True)
        callbacks.append(checkpoint)
        
    
    hist =  model.fit(train.make_tfdata(),
                      epochs=flags.epoch,
                      validation_data=test.make_tfdata(),
                      batch_size=flags.batch,
                      callbacks=callbacks,                  
                      steps_per_epoch=train.steps_per_epoch,
                      validation_steps =test.steps_per_epoch,
                      verbose=hvd.rank() == 0,
                      )
    
