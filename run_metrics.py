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
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
import yaml
from scipy.interpolate import griddata

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

colorlist = ['blue','orange','green','red','purple','brown','darkgrey','cyan','deepskyblue','dimgrey','lightcoral','teal','darkviolet','magenta','chocolate','dodgerblue','olive','lawngreen','paleturquoise','darkred']

with open("./pf.json") as jsonfile:
    payload = json.load(jsonfile)
    weight = payload['weight']
    features = payload['features']
    altfeatures = payload['altfeatures']
    cut = payload['cut']
    ss = payload['ss_vars']
    label = payload['!decayType']

feature_range = {
        "fj_msd":[40.,215.],
        "fj_pt":[300.,700.],
        }

mincut = -1.
maxcut = 1.
stepsize = 0.05
ddt_pts = [0.01,0.05,0.1,0.5]

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

def load(iFile_sig, iFile_bkg, iFileNpy='',iNparts=20,iNSVs=5):
    h5File_sig = h5py.File(iFile_sig)
    h5File_bkg = h5py.File(iFile_bkg)
    treeArray_sig = h5File_sig['deepDoubleTau'][()]
    treeArray_bkg = h5File_bkg['deepDoubleTau'][()]
    #treeArray_bkg = treeArray_bkg[:10000,:]
    treeArray_bkg = treeArray_bkg[:1500000,:]
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
    selSVfeatures = []
    for iVar in altfeatures:
        for i0 in range(iNSVs):
            selSVfeatures.append(iVar+str(i0))
    print(features_labels_df_sig[label])
    print(features_labels_df_bkg[label])
    features_df_sig        = features_labels_df_sig[selPartfeatures].values
    features_df_bkg        = features_labels_df_bkg[selPartfeatures].values
    features_sv_df_sig        = features_labels_df_sig[selSVfeatures].values
    features_sv_df_bkg        = features_labels_df_bkg[selSVfeatures].values
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
    print(features_sv_df_sig.shape)
    features_sv_df_sig = features_sv_df_sig.reshape(-1,iNSVs,len(altfeatures))
    features_sv_df_bkg = features_sv_df_bkg.reshape(-1,iNSVs,len(altfeatures))
    print(features_sv_df_sig.shape)
    features_val = np.concatenate([features_df_sig, features_df_bkg])
    features_alt_val = np.concatenate([features_sv_df_sig, features_sv_df_bkg])
    labels_val = np.concatenate([labels_sig, labels_bkg])
    feat_val = np.concatenate([feat_val_sig, feat_val_bkg])

    print(features_val)
    # split into random test and train subsets 
    X_train_val, X_test, Xalt_train_val, Xalt_test, y_train_val, y_test, feat_train, feat_test = train_test_split(features_val, features_alt_val, labels_val, feat_val, test_size=0.2, random_state=42)
    #scaler = preprocessing.StandardScaler().fit(X_train_val)
    #X_train_val = scaler.transform(X_train_val)
    #X_test      = scaler.transform(X_test)
    return X_train_val, X_test, Xalt_train_val, Xalt_test, y_train_val, y_test, feat_train, feat_test

def conditional_loss_function(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)*(1-y_true[:,0])

def plotNNResponse(data,labels,savename):
    plt.clf()
    bins=100
    for j in range(len(data)):
      plt.hist(data[j],bins,log=False,histtype='step',normed=True,label=labels[j],fill=False,range=(-1.,1.))
    plt.legend(loc='best')
    plt.xlabel('NeuralNet Response')
    plt.ylabel('Number of events (normalized)')
    plt.title('NeuralNet applied to test samples')
    plt.savefig("deepDoubleTau_%s_disc.pdf"%savename)
    plt.yscale('log')
    plt.savefig("deepDoubleTau_%s_disc_log.pdf"%savename)
    plt.yscale('linear')
    #plt.show()

def doDDT(data,feats,nbins,xbins,ybins,percents,savename):

    rho_arr = 2.*np.log(np.divide(feats[0][:,1],feats[0][:,0]))
    pt_arr = feats[0][:,0]

    xstep = (xbins[-1]-xbins[0])/float(nbins-1)
    ystep = (ybins[-1]-ybins[0])/float(nbins-1)
    binix = np.digitize(rho_arr,np.arange(xbins[0],xbins[-1]+xstep,xstep))
    biniy = np.digitize(pt_arr,np.arange(ybins[0],ybins[-1]+ystep,ystep))

    ddt_list = []
    base_list = []
    for x in range(nbins):
        for y in range(nbins):
            tmpdata = np.array(data[0])[(binix==x) & (biniy==y)]
            if (len(tmpdata)>0): ddt_list.append(np.quantile(tmpdata,1.-np.array(percents)))
            else: ddt_list.append(np.zeros(len(percents)))
            base_list.append([xbins[0]+x*xstep,ybins[0]+y*ystep])
    ddt_arr = np.array(ddt_list)
    base_grid = np.array(base_list)

    x_arr = np.array([xbins[a] for a in range(len(xbins)-1) for b in range(len(ybins)-1)])
    y_arr = np.array([ybins[b] for a in range(len(xbins)-1) for b in range(len(ybins)-1)])

    #base_grid = np.stack((np.arange(xbins[0],xbins[-1]+xstep,xstep),np.arange(ybins[0],ybins[-1]+ystep,ystep)),axis=-1)
    ddt_smooth = [griddata(base_grid, ddt_arr[:,ip], np.stack((x_arr, y_arr),axis=-1), method='cubic').clip(0.) for ip in range(len(percents))]
    print(ddt_smooth)

    for ip in range(len(percents)):
        plt.clf()
        plt.hist2d(x_arr,y_arr,weights=ddt_smooth[ip],bins=[xbins,ybins])
        plt.title('DDT Map ({}%)'.format(int(percents[ip]*100.)))
        plt.ylabel(r'$p_{T}$')
        plt.xlabel(r'$\rho = log(p_{T}^2/m_{SD}^2)$')
        plt.colorbar()
        plt.savefig("deepDoubleTau_%s_ddt_surface_%d.pdf"%(savename,int(percents[ip]*100.)))

    return ddt_smooth

def applyDDT(data,feats,xbins,ybins,themap):

    data_ddt = []
    for d in range(len(data)):

        rho_arr = 2.*np.log(np.divide(feats[d][:,1],feats[d][:,0]))
        pt_arr = feats[d][:,0]
        binix = np.digitize(rho_arr,xbins)
        biniy = np.digitize(pt_arr,ybins)
        tmpddt = np.zeros(data[d].shape)

        for x in range(len(xbins)-1):
            for y in range(len(ybins)-1):
                if (len(data[d][((binix==x) & (biniy==y))])>0): tmpddt[((binix==x) & (biniy==y))] = data[d][((binix==x) & (biniy==y))] - themap[(x*(len(ybins)-1))+y]
        data_ddt.append(tmpddt)

    return data_ddt

def plotFeatResponse(data,feats,labels, nnout_cuts, savename, ddt_grp = None):

    if (ddt_grp is not None): 
      nnout_cuts = np.zeros(len(ddt_pts))
    feats_pass = []
    for j in range(len(data)):
      bufpass = []
      for c in range(len(nnout_cuts)-1):
        tmppass = []
        tmpdata = data
        if (ddt_grp is not None):
          tmpdata = applyDDT(data,feats,ddt_grp[1],ddt_grp[2],ddt_grp[0][c])
          for x in range(len(data[j])-1):
            if (tmpdata[j][x]>0.): tmppass.append(feats[j][x])
        else:
          for x in range(len(data[j])-1):
            if (c<len(nnout_cuts)-2):
              if (tmpdata[j][x]>=nnout_cuts[c] and tmpdata[j][x]<nnout_cuts[c+1]): tmppass.append(feats[j][x])
            else:
              if (tmpdata[j][x]>=nnout_cuts[c] and tmpdata[j][x]<=nnout_cuts[c+1]): tmppass.append(feats[j][x])
        
        print("p = %i  -> %i"%(c,len(tmppass)))
        bufpass.append(tmppass)
      feats_pass.append(bufpass)
    #[channel][cuts][entries][features]

    for fi in range(len(features_to_plot)):
      plt.clf()
      nbins=25

      fig, ax = plt.subplots(1, 1)

      feat_range = [0.,1.]
      if (features_to_plot[fi] in feature_range): feat_range = feature_range[features_to_plot[fi]]

      for j in range(len(data)):
        if (j==0):
          for ci in range(len(feats_pass[j])):
            doPlot = False
            temp = np.array(feats_pass[j][ci])
            temp = temp.clip(min=0)
            temp = np.reshape(temp,(-1,len(features_to_plot)))
            pfstr = ', '
            if (ddt_grp is None): pfstr = pfstr + '{}%'.format(int(float(ci*stepsize)*100.))
            else: pfstr = pfstr + '{}%'.format(int(ddt_pts[ci]*100.))
            stylestr = "dashed"
            if (len(temp[:,fi])>0):
                ax.hist(temp[:,fi],nbins,log=False,histtype='step',normed=True,linestyle=stylestr,
                        label=labels[j]+pfstr,fill=False,range=feat_range,color=colorlist[ci])
        else:
          doPlot = False
          temp = np.array(feats[j])
          #print(temp.shape)
          temp = temp[:,fi]
          temp = temp.clip(min=0.)
          if (features_to_plot[fi] in feature_range): feat_range = feature_range[features_to_plot[fi]]
          pfstr = ''
          stylestr = "solid"
          if (len(temp)>0):
              ax.hist(temp,nbins,log=False,histtype='step',normed=True,linestyle=stylestr, 
                      label=labels[j]+pfstr,fill=False,range=feat_range,color='black')
      ax.legend(loc='best',ncol=2)
      ax.set_xlabel(features_to_plot[fi])
      ax.set_ylabel('Number of events (normalized)')
      ax.set_title('Neural Net')
      plt.savefig("deepDoubleTau_"+savename+"_"+features_to_plot[fi]+".pdf")
      plt.yscale('log')
      plt.savefig("deepDoubleTau_"+savename+"_"+features_to_plot[fi]+"_log.pdf")
      plt.yscale('linear')
      #plt.show()

def plotROC(truth, scores,labels,savename):
    plt.clf()
    for j in range(len(truth)):
        x,y,_ = roc_curve(truth[j],scores[j])
        auc = roc_auc_score(truth[j],scores[j])
        plt.plot(x,y,label='{}, AUC = {:.2f}'.format(labels[j],auc))
    plt.legend(loc='lower right')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig("deepDoubleTau_%s_roc.pdf"%savename)
    #plt.show()

def computePercentiles(data):
    
    tmp = np.quantile(data,np.arange(stepsize, 1., stepsize))
    tmpl = [mincut]
    for x in tmp:
        tmpl.append(x)
    tmpl.append(maxcut)
    return tmpl

def test(models,X_test,Xalt_test,Y_test,feat_test, savename, ddt_map=None):
    model = models['classifier']
    subsamples = [ 
                  [0,0], #bkg
                  [1,1] ] #h tautau hadhad
    labels     = ["Bkg","htt"]
    roclabels  =       ["deepDoubleTau"]
    roclabels = roclabels + ss
    response_tests   = []
    response_preds   = []
    feat_preds       = []
    roc_preds        = []
    roc_true         = []
    i0=0
    print('in test')
    for subsample in subsamples:
        print(subsample)
        ids=np.logical_and(Y_test==subsample[0],Y_test==subsample[1])
        tmpdata = [X_test[ids],Xalt_test[ids]]
        tmpfeat = feat_test[ids]
        tmppred = model.predict(tmpdata)
        print('\t',len(tmppred))
        response_tests.append(tmpdata)
        response_preds.append(tmppred)
        feat_preds.append(tmpfeat)
        if i0 > 0 and len(tmpdata[0]) > 0:
            roc_true.append([])
            roc_true[-1].extend(np.zeros(len(response_tests[0][0] )))
            roc_true[-1].extend(np.ones (len(tmpdata[0])))
            roc_preds.append([])
            roc_preds[-1].extend(response_preds[0])
            roc_preds[-1].extend(tmppred)

        for fi in range(len(weight),len(weight)+len(ss)):
            tmppred = np.array(tmpfeat)[:,fi]
            print('\t',len(tmppred))
            if i0 > 0 and len(tmpdata[0]) > 0:
                roc_true.append([])
                roc_true[-1].extend(np.zeros(len(response_tests[0][0] )))
                roc_true[-1].extend(np.ones (len(tmpdata[0])))
                roc_preds.append([])
                roc_preds[-1].extend(response_preds[0])
                roc_preds[-1].extend(tmppred)

        i0=i0+1

    print('now plotting feats')
    nnout_cuts = computePercentiles(response_preds[labels.index("Bkg")])
    print(nnout_cuts)
    print(nnout_cuts[1:-1])
    plotFeatResponse(response_preds, feat_preds, labels, nnout_cuts, savename)

    binsx = np.arange(-5.5,-1.98,0.02)
    binsy = np.arange(300.,1005.,5.)
    nbins_ddt = 10

    if (ddt_map is None):
        print('now computing/plotting DDT')
        ddt_map = doDDT(response_preds,feat_preds,nbins_ddt,binsx,binsy,np.array(ddt_pts),savename)

    #print(ddt_map)
    #for ir,r in enumerate(response_preds):
    #    print(labels[ir])
    #    print(np.histogram(r))
    print('now plotting resp')
    plotNNResponse(response_preds,labels,savename)
    print('now plotting ddt feats')
    plotFeatResponse(response_preds, feat_preds, labels, None, savename+"_ddt",[ddt_map,binsx,binsy])
    print('now plotting roc')
    plotROC(roc_true, roc_preds, roclabels, savename) 

    return ddt_map


if __name__ == "__main__":
    dtype = 'hadmu'
    X_train,X_test,Xalt_train,Xalt_test,Y_train,Y_test,feat_train,feat_test = load('./GluGluHToTauTau_user_%s.z'%dtype,'./QCD.z')
    models = {}
    for m in ['classifier']:
        json_file = open('model_'+dtype+'_'+str(m)+'.json', 'r')
        model_json = json_file.read()
        models[m] = model_from_json(model_json)
        models[m].load_weights("model_"+dtype+"_"+str(m)+".h5")
    qcd_ddt = test(models,X_test,Xalt_test,Y_test,feat_test,dtype+"_QCD")
    X_train,X_test,Xalt_train,Xalt_test,Y_train,Y_test,feat_train,feat_test = load('./GluGluHToTauTau_user_%s.z'%dtype,'./TTbar.z')
    test(models,X_test,Xalt_test,Y_test,feat_test,dtype+"_TTbar",ddt_map=qcd_ddt)
