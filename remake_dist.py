import os,sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
#import keras
import numpy as np
#from keras import backend as K
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()
from optparse import OptionParser
import pandas as pd
import h5py
import json
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('agg')
#%matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization, GRU
from tensorflow.keras.models import Model 
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
import yaml

fColors = {
'black'    : (0.000, 0.000, 0.000), # hex:000000
'blue'     : (0.122, 0.467, 0.706), # hex:1f77b4
'orange'   : (1.000, 0.498, 0.055), # hex:ff7f0e
'green'    : (0.173, 0.627, 0.173), # hex:2ca02c
'red'      : (0.839, 0.153, 0.157), # hex:d62728
'purple'   : (0.580, 0.404, 0.741), # hex:9467bd
'brown'    : (0.549, 0.337, 0.294), # hex:8c564b
'darkgrey' : (0.498, 0.498, 0.498), # hex:7f7f7f
'olive'    : (0.737, 0.741, 0.133), # hex:bcbd22
'cyan'     : (0.090, 0.745, 0.812)  # hex:17becf
}

colorlist = ['blue','orange','green','red','purple','brown','darkgrey','cyan']

with open("./pf.json") as jsonfile:
    payload = json.load(jsonfile)
    weight = payload['weight']
    features = payload['features']
    altfeatures = payload['altfeatures']
    cut = payload['cut']
    ss = payload['ss_vars']
    label = payload['!decayType']

# columns declared in file
lColumns = weight + ss
nparts = 30
lPartfeatures = []
for iVar in features:
    for i0 in range(nparts):
        lPartfeatures.append(iVar+str(i0))
nsvs = 5
lSVfeatures = []
for iVar in altfeatures:
    for i0 in range(nsvs):
        lSVfeatures.append(iVar+str(i0))
lColumns = lColumns + lPartfeatures + lSVfeatures + [label]

features_to_plot = weight + ss

fill_factor = 5

def turnon(iD,iTrainable,iOther=0):
    i0 = -1
    for l1 in iD.layers:
        i0=i0+1
        if iOther != 0 and l1 in iOther.layers:
            continue
        try:
            l1.trainable = iTrainable
        except:
            print("trainableErr",layer)

def remake(iFiles_sig, iFiles_bkg, iFile_out):


    features_labels_df_sig = pd.DataFrame(columns=lColumns)

    for sig in iFiles_sig:
        h5File_sig = h5py.File(sig)
        treeArray_sig = h5File_sig['deepDoubleTau'][()]
        print(treeArray_sig.shape)
        tmp_sig = pd.DataFrame(treeArray_sig,columns=lColumns)
        features_labels_df_sig = pd.concat([features_labels_df_sig,tmp_sig])

    sighist,_x,_y = np.histogram2d(features_labels_df_sig[weight[0]],features_labels_df_sig[weight[1]],bins=20,range=np.array([[300.,800.],[40.,240.]]))
    print(np.sum(sighist))

    remade_df_bkg = pd.DataFrame(columns=lColumns)
    for bkg in iFiles_bkg:
        h5File_bkg = h5py.File(bkg)
        treeArray_bkg = h5File_bkg['deepDoubleTau'][()]
        print(treeArray_bkg.shape)
        tmp_bkg = pd.DataFrame(treeArray_bkg,columns=lColumns)

        for ix in range(len(_x)-1):
            for iy in range(len(_y)-1):
                remade_df_bkg = pd.concat([remade_df_bkg,tmp_bkg[((tmp_bkg[weight[0]] >= _x[ix]) & (tmp_bkg[weight[0]] < _x[ix+1]) & (tmp_bkg[weight[1]] >= _y[iy]) & (tmp_bkg[weight[1]] < _y[iy+1]))].head(int(sighist[ix,iy])*fill_factor)], ignore_index = True)

    bkghist,_,_ = np.histogram2d(remade_df_bkg[weight[0]],remade_df_bkg[weight[1]],bins=20,range=np.array([[300.,800.],[40.,240.]]))
    print(np.nan_to_num(np.divide(bkghist,sighist)))

    arr = np.concatenate([features_labels_df_sig.values,remade_df_bkg.values],axis=0)
    print(arr.shape)
    # open HDF5 file and write dataset
    h5File = h5py.File(iFile_out,'w')
    h5File.create_dataset('deepDoubleTau', data=arr,  compression='lzf')
    h5File.close()
    del h5File


if __name__ == "__main__":
    remake(['./GluGluHToTauTau_user_hadel.z'],['./QCD.z','./TTbar.z'],'./comb_distcut%i_hadel.z'%fill_factor)
    remake(['./GluGluHToTauTau_user_hadmu.z'],['./QCD.z','./TTbar.z'],'./comb_distcut%i_hadmu.z'%fill_factor)
    remake(['./GluGluHToTauTau_user_hadhad.z'],['./QCD.z','./TTbar.z'],'./comb_distcut%i_hadhad.z'%fill_factor)

