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
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization, GRU, Add
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

bin_dict = {
        "fj_pt":np.arange(300.,825.,25.),
        "fj_msd":np.arange(40.,215.,5.),
        }

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

def load_comb(iFile, iFileNpy='',iNparts=20,iNSVs=5):
    h5File = h5py.File(iFile)
    treeArray = h5File['deepDoubleTau'][()]
    print(treeArray.shape)

    features_labels_df = pd.DataFrame(treeArray,columns=lColumns)

    idconv = {211.:1, 13.:2,  22.:3,  11.:4, 130.:5, 1.:6, 2.:7, 3.:8, 4.:9,
            5.:10, -211.:1, -13.:2,
            -11.:4, -1.:-6, -2.:7, -3.:8, -4.:9, -5.:10, 0.:0}
    nIDs = 33
    for i0 in range(nparts):
        features_labels_df['PF_id'+str(i0)] = features_labels_df['PF_id'+str(i0)].map(idconv)
    selPartfeatures = []
    for iVar in features:
        for i0 in range(iNparts):
            selPartfeatures.append(iVar+str(i0))
    selSVfeatures = []
    for iVar in altfeatures:
        for i0 in range(iNSVs):
            selSVfeatures.append(iVar+str(i0))
    features_df        = features_labels_df[selPartfeatures].values
    features_sv_df        = features_labels_df[selSVfeatures].values
    labels          = features_labels_df[label]
    #features_val       = features_df.values
    feat_val           = features_labels_df[features_to_plot].values

    for p in selPartfeatures:
        if (features_labels_df[p].isna().sum()>0): print(p,"found nan!!")

    #if iFileNpy!='':
    #    features_2df = np.load(iFileNpy)
    #else:
    print(features_df.shape)
    features_df = features_df.reshape(-1,iNparts,len(features))
    print(features_df.shape)
    print(features_sv_df.shape)
    features_sv_df = features_sv_df.reshape(-1,iNSVs,len(altfeatures))
    print(features_sv_df.shape)
    features_val = features_df
    features_sv_val = features_sv_df
    labels_val = labels
    feat_val = feat_val

    print(features_val)
    # split into random test and train subsets 
    X_train_val, X_test, Xalt_train_val, Xalt_test, y_train_val, y_test, feat_train, feat_test = train_test_split(features_val, features_sv_val, labels_val, feat_val, test_size=0.2, random_state=42)
    #scaler = preprocessing.StandardScaler().fit(X_train_val)
    #X_train_val = scaler.transform(X_train_val)
    #X_test      = scaler.transform(X_test)
    return X_train_val, X_test, Xalt_train_val, Xalt_test, y_train_val, y_test, feat_train, feat_test


def load(iFile_sig, iFile_bkg, iFileNpy='',iNparts=20):
    h5File_sig = h5py.File(iFile_sig)
    h5File_bkg = h5py.File(iFile_bkg)
    treeArray_sig = h5File_sig['deepDoubleTau'][()]
    treeArray_bkg = h5File_bkg['deepDoubleTau'][()]
    #treeArray_bkg = treeArray_bkg[:10000,:]
    print(treeArray_sig.shape)
    print(treeArray_bkg.shape)

    features_labels_df_sig = pd.DataFrame(treeArray_sig,columns=lColumns)
    features_labels_df_bkg = pd.DataFrame(treeArray_bkg,columns=lColumns)

    idconv = {211.:1, 13.:2,  22.:3,  11.:4, 130.:5, 1.:6, 2.:7, 3.:8, 4.:9,
            5.:10, -211.:1, -13.:2,
            -11.:4, -1.:-6, -2.:7, -3.:8, -4.:9, -5.:10, 0.:0}
    nIDs = 33
    for i0 in range(nparts):
        features_labels_df_sig['PF_id'+str(i0)] = features_labels_df_sig['PF_id'+str(i0)].map(idconv)
        features_labels_df_bkg['PF_id'+str(i0)] = features_labels_df_bkg['PF_id'+str(i0)].map(idconv)
    selPartfeatures = []
    for iVar in features:
        for i0 in range(iNparts):
            selPartfeatures.append(iVar+str(i0))
    features_df_sig        = features_labels_df_sig[selPartfeatures].values
    features_df_bkg        = features_labels_df_bkg[selPartfeatures].values
    labels_sig        = features_labels_df_sig[label]
    labels_bkg        = features_labels_df_bkg[label]
    #features_val       = features_df.values
    feat_val_sig           = features_labels_df_sig[features_to_plot].values
    feat_val_bkg           = features_labels_df_bkg[features_to_plot].values

    for p in selPartfeatures:
        if (features_labels_df_sig[p].isna().sum()>0): print(p,"found nan!!")
        if (features_labels_df_bkg[p].isna().sum()>0): print(p,"found nan!!")

    #if iFileNpy!='':
    #    features_2df = np.load(iFileNpy)
    #else:
    print(features_df_sig.shape)
    features_df_sig = features_df_sig.reshape(-1,iNparts,len(features))
    features_df_bkg = features_df_bkg.reshape(-1,iNparts,len(features))
    print(features_df_sig.shape)
    features_val = np.concatenate([features_df_sig, features_df_bkg])
    labels_val = np.concatenate([labels_sig, labels_bkg])
    feat_val = np.concatenate([feat_val_sig, feat_val_bkg])

    print(features_val)
    # split into random test and train subsets 
    X_train_val, X_test, y_train_val, y_test, feat_train, feat_test = train_test_split(features_val, labels_val, feat_val, test_size=0.2, random_state=42)
    #scaler = preprocessing.StandardScaler().fit(X_train_val)
    #X_train_val = scaler.transform(X_train_val)
    #X_test      = scaler.transform(X_test)
    return X_train_val, X_test, y_train_val, y_test, feat_train, feat_test

def conditional_loss_function(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)*(1-y_true[:,0])

def model(Inputs,Inputs_alt,X_train,Xalt_train,Y_train,NPARTS=20,NSV=5):
    CLR=0.001
    L1R=0.00001
    print(Inputs)
    print(Inputs_alt)

    gru = GRU(100,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base',activity_regularizer=l1(L1R))(Inputs)
    dense   = Dense(100, activation='relu',activity_regularizer=l1(L1R))(gru)
    norm    = BatchNormalization(momentum=0.6, name='dense4_bnorm')  (dense)

    gru_alt = GRU(100,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base_alt',activity_regularizer=l1(L1R))(Inputs_alt)
    dense_alt   = Dense(100, activation='relu',activity_regularizer=l1(L1R))(gru_alt)
    norm_alt    = BatchNormalization(momentum=0.6, name='dense4_bnorm_alt')  (dense_alt)

    added = Add()([norm, norm_alt])

    dense   = Dense(50, activation='relu',activity_regularizer=l1(L1R))(added)
    norm    = BatchNormalization(momentum=0.6, name='dense5_bnorm')  (dense)
    dense   = Dense(20, activation='relu',activity_regularizer=l1(L1R))(norm)
    dense   = Dense(10, activation='relu',activity_regularizer=l1(L1R))(dense)
    out     = Dense(1, activation='sigmoid',activity_regularizer=l1(L1R))(norm)
    
    classifier = Model(inputs=[Inputs,Inputs_alt], outputs=[out])
    lossfunction = 'binary_crossentropy'
    classifier.compile(loss=[lossfunction], optimizer=Adam(CLR), metrics=['accuracy'])
    models={'classifier' : classifier}

    return models

def train(models,X_train,Xalt_train,Y_train,feat_train,doReweight=False):
    NEPOCHS=20
    Obatch_size=1000

    history = {}
    if (doReweight):
        sighist,_x,_y = np.histogram2d(feat_train[(Y_train.values==1),0],feat_train[(Y_train.values==1),1],bins=20,range=np.array([[300.,800.],[40.,240.]]))
        bkghist,_,_ = np.histogram2d(feat_train[(Y_train.values==0),0],feat_train[(Y_train.values==0),1],bins=[_x,_y])
        ratio_sb = np.nan_to_num(np.divide(sighist,bkghist))

        weights = np.ones(len(Y_train.values))
        binix = np.digitize(feat_train[:,0],_x)
        biniy = np.digitize(feat_train[:,1],_y)
        for i in range(len(weights)):
            if (Y_train.values[i]==0):
                if (binix[i]>0 and binix[i]<len(_x)-1 and biniy[i]>0 and biniy[i]<len(_y)-1): weights[i] = ratio_sb[binix[i],biniy[i]]

        print(weights)

        history["classifier"] = models['classifier'].fit([X_train,Xalt_train],
            Y_train,epochs=NEPOCHS,verbose=1,batch_size=Obatch_size,
            validation_split=0.2,sample_weight=weights)
    else: history["classifier"] = models['classifier'].fit([X_train,Xalt_train],
            Y_train,epochs=NEPOCHS,verbose=1,batch_size=Obatch_size,
            validation_split=0.2)
     
    return history

if __name__ == "__main__":
    dtype = "hadmu"
    #X_train,X_test,Y_train,Y_test,feat_train,feat_test = load('./GluGluHToTauTau_user_%s.z'%dtype,'./QCD.z')
    X_train,X_test,Xalt_train,Xalt_test,Y_train,Y_test,feat_train,feat_test = load_comb('./comb_distcut5_%s.z'%dtype)

    for iw,w in enumerate(weight):
        plt.clf()
        plt.hist(feat_train[(Y_train.values==0),iw],bin_dict[w],log=False,histtype='step',normed=True,fill=False,label='bkg')
        plt.hist(feat_train[(Y_train.values==1),iw],bin_dict[w],log=False,histtype='step',normed=True,fill=False,label='sig')
        plt.legend(loc='best')
        plt.xlabel(w)
        plt.ylabel('arb.')
        plt.savefig("%s_%s.pdf"%(w,dtype))
        plt.yscale('log')
        plt.savefig("%s_%s_log.pdf"%(w,dtype))
        plt.yscale('linear')

    inputvars=Input(shape=X_train.shape[1:], name='input')
    inputvars_alt=Input(shape=Xalt_train.shape[1:], name='altinput')
    models = model(inputvars,inputvars_alt,X_train,Xalt_train,Y_train)
    for m in models:
        print(str(m), models[m])
    history = train(models,X_train,Xalt_train,Y_train,feat_train)
    #print(len(Y_test),' vs ',sum(Y_test))
    #test(models,X_test,Y_test,feat_test)
    for m in models:
        model_json = models[m].to_json()
        with open("model_"+dtype+"_"+str(m)+".json", "w") as json_file:
            json_file.write(model_json)
        models[m].save_weights("model_"+dtype+"_"+str(m)+".h5")

        plt.clf() 
        print(history[str(m)].history.keys())
        plt.plot(history[str(m)].history['acc'])
        plt.plot(history[str(m)].history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("%s_%s_accuracy.pdf"%(dtype,str(m)))
        plt.clf() 
        plt.plot(history[str(m)].history['loss'])
        plt.plot(history[str(m)].history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("%s_%s_loss.pdf"%(dtype,str(m)))

