import json, yaml
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from sklearn.utils import shuffle
import tensorflow as tf
np.random.seed(0) #fix the seed to keep track of validation split

line_style = {
    'gluon_truth': '-',
    'gluon_gen': '-',
    'top_truth':'-',
    'top_gen':'-',
    'HV':'-',
}

colors = {
    'gluon_truth':'black',
    'gluon_gen':'#d95f02',
    'top_truth':'black',
    'top_gen':'#1b9e77',
    'HV':'#7570b3',
}


name_translate={
    'gluon_truth':'QCD',
    'gluon_gen':'QCD Gen,',
    'top_truth':'Top',
    'top_gen':'Top Gen.',
    'HV': "Z'",
}


def revert_npart(npart):
    #Revert the preprocessing to recover the particle multiplicity
    mean =  43.34036032
    std = 18.63108734
    x = npart*std + mean
    return np.round(x).astype(np.int32)


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    # import mplhep as hep
    # hep.set_style(hep.style.CMS)
    
    # hep.style.use("CMS") 

    
def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

def SetFig(xlabel,ylabel):
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    
    ax0.minorticks_on()
    return fig, ax0

    
def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='gen'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            ratio = 100*np.divide(np.mean(feed_dict[reference_name],0)-np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0))
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
                
        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=16,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-100,100])

    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.1f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)

def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,
                xlabel='',ylabel='',
                reference_name='Geant',
                logy=False,binning=None,
                fig = None, gs = None,
                plot_ratio= True,
                idx = None,
                label_loc='best'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    if fig is None:
        fig,gs = SetGrid(plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)
        
    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),20)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    maxy = np.max(reference_hist)
    for ip,plot in enumerate(feed_dict.keys()):
        dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=name_translate[plot],linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
        if plot_ratio:
            if reference_name!=plot:
                ratio = 100*np.divide(reference_hist-dist,reference_hist)
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    ax0.legend(loc=label_loc,fontsize=14,ncol=2)
    ax0.set_ylim(top=2.1*maxy)
    if logy:
        ax0.set_yscale('log')



    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
        plt.ylabel('Difference. (%)')
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-100,100])
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0)
    
    return fig,gs, binning


def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)



class DataLoader():
    def __init__(self, path, batch_size=512,rank=0,size=1):
    
        self.path = path
        self.X = h5.File(self.path,'r')['data'][rank::size]
        self.jet = h5.File(self.path,'r')['jet'][rank::size]
        self.mask = self.X[:,:,2]!=0

        self.batch_size = batch_size
        self.nevts = h5.File(self.path,'r')['jet'][:].shape[0]
        self.num_part = self.X.shape[1]

        self.num_feat = self.X.shape[2]
        self.num_jet = self.jet.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

        self.mean_part = [ 9.06004446e-05, -4.00336919e-05 , 1.42354690e+00]
        self.std_part = [0.21700346, 0.21835248, 1.31198715]
        self.mean_global =  [95.44569975, 43.34036032]
        self.std_global  = [50.00375775, 18.63108734]

    def combine(self,datasets):
        self.label = np.zeros((self.X.shape[0],1))
        for dataset in datasets:
            self.nevts += dataset.nevts
            self.X = np.concatenate([self.X,dataset.X],0)
            self.mask = np.concatenate([self.mask,dataset.mask],0)
            self.jet = np.concatenate([self.jet,dataset.jet],0)
            self.label = np.concatenate([self.label,np.ones((dataset.X.shape[0],1))],0)


        self.X,self.mask,self.jet,self.label= shuffle(self.X,self.mask,self.jet,self.label)
        
        
    def make_tfdata(self):
        self.X = self.preprocess(self.X,self.mask).astype(np.float32)
        self.jet = self.preprocess_event(self.jet).astype(np.float32)

        training_data = {'input_particles':self.X,
                         'input_jets':self.jet,
                         'input_mask':self.mask.astype(np.float32),
                         }

                
        return training_data


    def data_from_file(self,file_path):
        with h5.File(file_path, 'r') as file:
            data_chunk = file['data'][:]
            mask_chunk = data_chunk[:, :, 2] != 0
            
            jet_chunk = file['jet'][:]

            data_chunk = self.preprocess(data_chunk, mask_chunk)
            jet_chunk = self.preprocess_event(jet_chunk)
            
        return data_chunk,jet_chunk,mask_chunk


    def preprocess(self,x,mask):
        return mask[:,:, None]*(x-self.mean_part)/self.std_part

    def preprocess_event(self,x):        

        return (x-self.mean_global)/self.std_global

    def revert_preprocess(self,x,mask):                
        new_part = mask[:,:, None]*(x*self.std_part + self.mean_part)
        return  new_part

    def revert_preprocess_jet(self,x):
        new_x = self.std_global*x+self.mean_global
        return new_x



