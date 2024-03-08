import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf
from tensorflow import keras
import sys
import horovod.tensorflow.keras as hvd
import pickle
from GSGM import GSGM
import utils


import matplotlib.pyplot as plt
utils.SetStyle()




def plot(jet1,jet2,nplots,title,plot_folder,sample='gluon',var_names=[]):
        
    for ivar in range(nplots):        
        feed_dict = {
            '{}_truth'.format(sample):jet1[:,ivar],
            '{}_gen'.format(sample):  jet2[:,ivar]
        }
            

        fig,gs,binning = utils.HistRoutine(feed_dict,xlabel="{}".format(var_names[ivar]),
                                           plot_ratio=True,
                                           reference_name='{}_truth'.format(sample),
                                           ylabel= 'Normalized entries')
        ax0 = plt.subplot(gs[0])     
        fig.savefig('{}/LL_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')


def likelihood_data(flags,sample_name):
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    test = utils.DataLoader(os.path.join(flags.folder, sample_name),
                            rank=hvd.rank(),size=hvd.size())

    model = GSGM(num_feat = test.num_feat,
                 num_jet = test.num_jet,
                 num_part=test.num_part)

    
    model_name = '../checkpoints/{}.weights.h5'.format(flags.dataset)
    model.load_weights('{}'.format(model_name))

    data = test.make_tfdata()

    ll_part,ll_jet,n_d,n_j = model.get_likelihood(data['input_particles'],
                                          data['input_jets'],
                                          data['input_mask'][:,:,None])
    
    
    ll_particles = hvd.allgather(tf.constant(ll_part)).numpy()
    ll_jets = hvd.allgather(tf.constant(ll_jet)).numpy()
    normal_data = hvd.allgather(tf.constant(n_d)).numpy()
    normal_jet = hvd.allgather(tf.constant(n_j)).numpy()

    
    if hvd.rank()==0:
        with h5.File(sample_name.replace('.h5','_ll.h5'),"w") as h5f:
            dset = h5f.create_dataset("ll_data", data=ll_particles)
            dset = h5f.create_dataset("ll_jet", data=ll_jets)
            dset = h5f.create_dataset("normal_jet", data=normal_jet)
            dset = h5f.create_dataset("normal_data", data=normal_data)
            

def sample_data(flags,sample_name,nevts=100):
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    test = utils.DataLoader(os.path.join(flags.folder, '{}_val.h5'.format(flags.dataset)),
                            rank=hvd.rank(),size=hvd.size())    
    

    model = GSGM(num_feat = test.num_feat,
                 num_jet = test.num_jet,
                 num_part=test.num_part)

    
    model_name = '../checkpoints/{}.weights.h5'.format(flags.dataset)
    model.load_weights('{}'.format(model_name))

    p,j,n_d,n_j = model.generate(nevts)
    p = test.revert_preprocess(p,p[:,:,2]!=0)
    j = test.revert_preprocess_jet(j)
    
    particles_gen = hvd.allgather(tf.constant(p)).numpy()
    jets_gen = hvd.allgather(tf.constant(j)).numpy()
    normal_data = hvd.allgather(tf.constant(n_d)).numpy()
    normal_jet = hvd.allgather(tf.constant(n_j)).numpy()
    
    if hvd.rank()==0:
        with h5.File(sample_name,"w") as h5f:
            dset = h5f.create_dataset("data", data=particles_gen)
            dset = h5f.create_dataset("jet", data=jets_gen)
            dset = h5f.create_dataset("normal_jet", data=normal_jet)
            dset = h5f.create_dataset("normal_data", data=normal_data)
            

def get_generated_particles(sample_name):
    with h5.File(sample_name,"r") as h5f:
        jets_gen = h5f['jet'][:]
        particles_gen = h5f['data'][:,:,:3]        

    mask_gen = particles_gen[:,:,2]!=0
    particles_gen = particles_gen*mask_gen[:,:,None]
    return jets_gen, particles_gen


def get_gaussians(sample_name):
    with h5.File(sample_name,"r") as h5f:
        jets_gen = h5f['normal_jet'][:]
        particles_gen = h5f['normal_data'][:,:,:3]        

    return jets_gen, particles_gen


def get_from_file(test,nevts=-1):
    #Load eval samples for metric calculation
    particles,jets,mask = test.data_from_file(test.files[0])        
            
    particles = test.revert_preprocess(particles,mask)
    jets = test.revert_preprocess_jet(jets)
    #only keep the first 3 features
    if nevts<0:
        nevts = jets.shape[0]
        
    particles = particles[:nevts]*mask[:nevts,:,None]
    jets = jets[:nevts]

    return particles,jets
    


if __name__=='__main__':


    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--dataset", type="string", default="gluon", help="Dataset to load")
    parser.add_option("--folder", type="string", default="/global/cfs/cdirs/m3246/vmikuni/TOP/", help="Folder containing input files")
    
    #Model parameters    
    parser.add_option('--sample', action='store_true', default=False,help='Sample from trained model')
    parser.add_option('--likelihood', action='store_true', default=False,help='Estimate the likelihood from generated samples')
    parser.add_option("--plot_folder", type="string", default="../plots", help="Folder to save the outputs")
    parser.add_option("--ngen", type=int, default=1000, help="Number of samples to generate")
    
    (flags, args) = parser.parse_args()

    sample_name = os.path.join(flags.folder,"{}.h5".format(flags.dataset))
    
    if flags.sample:
        sample_data(flags,sample_name,flags.ngen)
    elif flags.likelihood:
        likelihood_data(flags,sample_name)
    else:        
        jets_gen, particles_gen = get_generated_particles(sample_name)
        print("Loading {} Generated Events with up to {} particles".format(jets_gen.shape[0], particles_gen.shape[1]))
        test = utils.DataLoader(os.path.join(flags.folder, '{}_val.h5'.format(flags.dataset)))    
        particles, jets = get_from_file(test)
                
        plot(jets,jets_gen,title='jet_{}'.format(flags.dataset),nplots=2,
             var_names=['Jet mass [GeV]','Jet Particle Multiplicity'],
             plot_folder=flags.plot_folder,sample=flags.dataset)
                    
        particles_gen=particles_gen.reshape((-1,3))
        mask_gen = particles_gen[:,2]!=0.
        particles_gen=particles_gen[mask_gen]
        particles=particles.reshape((-1,3))
        mask = particles[:,2]!=0.
        particles=particles[mask]
        
        plot(particles,particles_gen,
             title='part_{}'.format(flags.dataset),nplots=3,
             var_names = [r'Particle $\eta_{rel}$',r'Particle $\phi_{rel}$',r'Particle log(1.0 + p$_{T}$)'],
             plot_folder=flags.plot_folder,
             sample=flags.dataset)
        
        jets_init,part_init = get_gaussians(sample_name)
        jets_end,part_end = get_gaussians(sample_name.replace('.h5','_ll.h5'))
        print(jets_init-jets_end)
        plot(jets_init,jets_end,title='normal_jet_{}'.format(flags.dataset),nplots=2,
             var_names=['Jet Gaussian 1','Jet Gaussian 2'],
             plot_folder=flags.plot_folder,sample=flags.dataset)
        
        part_init = part_init.reshape((-1,3))
        mask_init = part_init[:,2]!=0.
        part_init = part_init[mask_gen]
        
        part_end = part_end.reshape((-1,3))
        mask_end = part_end[:,2]!=0.
        part_end = part_end[mask_end]

        plot(part_init,part_end,
             title='normal_part_{}'.format(flags.dataset),nplots=3,
             var_names = ['Particle Gaussian 1','Particle Gaussian 2','Particle Gaussian 3'],
             plot_folder=flags.plot_folder,
             sample=flags.dataset)
