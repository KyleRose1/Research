##################################################
# This script takes in training/ evaluation pat files
# or data from new idq runs, trains/evaluates
# a Keras ANN, and  produces ROC curve metrics
##################################################
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import Callback , EarlyStopping, ModelCheckpoint
from sklearn import metrics
from sklearn.preprocessing import normalize
from keras.callbacks import LambdaCallback
from keras.models import load_model
import os
import sys
import keras
print(keras.__version__)
#Data
run ='73'    #What data are we using.  
version='1'  #What version of this classifier are we using? 
             #(What are the model's hyperparameters - #layers,
             #nodes,activation function etc.)
sigma=3      #For error bars

DATA_train = '/home/rosek1/ANN/runs/%s/L-colated-1167536688-2253600.pat' % (run)
DATA_eval = '/home/rosek1/ANN/runs/%s/L-colated-1169787249-43200.pat' % (run)

# The path to save the model and important things
model_path_max = '/home/rosek1/ANN/runs/%s/KerasANN/version%s/model_max_%s' % (run,version,version)
model_path_end = '/home/rosek1/ANN/runs/%s/KerasANN/version%s/model_end_%s' % (run,version,version)
modelpath='/home/rosek1/ANN/runs/%s/KerasANN/version%s/model_run%svers%s.h5' % (run,version,run,version)
fap_path_max =  '/home/rosek1/ANN/runs/%s/KerasANN/version%s/fap_maxrun%svers%s' % (run,version,run,version)
eff_path_max =   '/home/rosek1/ANN/runs/%s/KerasANN/version%s/eff_maxrun%svers%s' % (run,version,run,version)
auc_path_max =   '/home/rosek1/ANN/runs/%s/KerasANN/version%s/auc_maxrun%svers%s' % (run,version,run,version)
eff_err_path_max='/home/rosek1/ANN/runs/%s/KerasANN/version%s/eff_err_maxrun%svers%s' % (run,version,run,version)
fap_err_path_max='/home/rosek1/ANN/runs/%s/KerasANN/version%s/fap_err_maxrun%svers%s' % (run,version,run,version)
predictions_path='/home/rosek1/ANN/runs/%s/KerasANN/version%s/predictionsrun%svers%s' % (run,version,run,version)
labels_path='/home/rosek1/ANN/runs/%s/KerasANN/version%s/labelsrun%svers%s' % (run,version,run,version)
#fap_path_end = '/home/rosek1/runs/%s/KerasANN/version%s/fap_end' % (run,version)
#eff_path_end = '/home/rosek1/runs/%s/KerasANN/version%s/eff_end' % (run,version)
#auc_path_end = '/home/rosek1/runs/%s/KerasANN/version%s/auc_end' % (run,version)

#Make Directories if they don't exist
paths=[modelpath,model_path_max,model_path_end,fap_path_max,eff_path_max,auc_path_max,eff_err_path_max,fap_err_path_max]
for path in paths:
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def read_data(DATA, randomize=False, level=False):
    '''
    read_data:
        This fucntion is a complete function that reads and seperates data into inputs
        and outputs for mla
    Inputs:
       -DATA: the directory path to the data
       -randomize: randomize the data, not necessary
    
    Outputs:
       -unclean: the label 0 or 1 associated with clean or unclean -- outputs of mla
       -quiver: the feature of each channel -- inputs of mla
       -gps_time: the gps time associated with each sample
    
    '''
    # Reading in the .pat file
    data = pd.read_csv(DATA, delim_whitespace=True, header=1,index_col=False)
    
    # Identify important columns
    feature_cols = [c for c in data.columns if c[:] != 'GPS_s' or 'GPS_ms' or 'signif' or 'SNR' or 'unclean']
    
    data = np.array(data.values)
    
    # removing any rows that have nan 
    nan_num = 0
    noNaNsense = []
    for z in range(len(data)):
        if np.isnan(data[z]).any() == True:
            print("NAN's in this pat file")
            noNaNsense=np.append(noNaNsense,[z],0)
    data = np.delete(data,noNaNsense,0)
    
    # Count the number of clean and glitch times
    unclean=0
    clean=0

    for b in range(len(data)):
        if data[[b],[4]] == 1:
                unclean+=1
        elif data[[b],[4]] == 0:
                clean+=1
    print("Length of Data: %d, Glitch: %d, Clean: %d" % (len(data),unclean,clean))
    
    if level == True:
        data = data[data[:,4].argsort()[::-1]]
        data = data[:2*unclean]
    
    # Option to randomize data
    if randomize == True:
        data = np.random.permutation(data)
    
    gps_s  = data[:,[0]]
    gps_ms = data[:,[1]] 
    labels = data[:,[4]]
    signif = data[:,[2]]

    dat_unnorm = np.delete(data[:],np.s_[0,1,2,3,4],1)
    inputs=normalize(dat_unnorm,norm='max',axis=0) #Divides each column by the max value in that column
    
    return inputs, labels, gps_s, gps_ms, feature_cols, clean, unclean, signif

###Extract training data
trainX, trainY, train_gps_s, train_gps_ms, train_fc, numcleantrain, numuncleantrain,trainsignif = read_data(DATA_train, randomize=False, level=False)
###Extract evaluation data
evalX, evalY, eval_gps_s, eval_gps_ms, eval_fc, numcleaneval, numuncleaneval, evalsignif = read_data(DATA_eval)

###Making sure the data sets match and won't break the ANN
assert len(trainX.T) == len(evalX.T)
assert len(trainX) == len(trainY)
assert len(evalX) == len(evalY)

n_inputs = len(trainX.T)
n_classes = len(trainY.T)

#Load an old model:
#model=load_model('/home/rosek/iDQ/runs/run24/trainmodelPostPickle.h5')

#Or make a new one:
###Neural Network
model=Sequential()
#Layers
model.add(Dense(units=100, activation='relu',input_dim=n_inputs))
#model.add(Dense(units=10000, activation='relu'))
#model.add(Dense(units=1000, activation='relu'))
#model.add(Dense(units=1000, activation='relu'))
#model.add(Dense(units=1000, activation='relu'))
#model.add(Dense(units=1000, activation='relu'))
#model.add(Dense(units=1000, activation='relu'))
#model.add(Dense(units=1000, activation='relu'))
#model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

optimizer1=keras.optimizers.Adam(lr=.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

###Model characteristics
model.compile(loss='binary_crossentropy',optimizer=optimizer1,metrics=['accuracy'])

#Stop training when:
callbacks=[EarlyStopping(monitor='val_loss',min_delta=0,patience=5,mode='auto'),ModelCheckpoint(modelpath,monitor='val_acc',verbose=0,save_best_only=True)]

###Train
print("Begin Training")
model.fit(trainX,trainY,epochs=1000,batch_size=32,\
validation_data=(evalX,evalY),callbacks=callbacks)
print("Done Training")

#Evaluate
print("Begin Evaluation")
y_score=model.predict_proba(evalX) #CHECK ME
np.savetxt(predictions_path,y_score)
np.savetxt(labels_path,evalY)

np.savetxt('/home/rosek1/ANN/runs/%s/KerasANN/version%s/ranks.txt' % (run,version),y_score)
np.savetxt('/home/rosek1/ANN/runs/%s/KerasANN/version%s/labels.txt' % (run,version),evalY)

fap, eff, thresh=metrics.roc_curve(evalY,y_score)

np.savetxt('/home/rosek1/ANN/runs/%s/KerasANN/version%s/faps.txt' % (run,version),fap)
np.savetxt('/home/rosek1/ANN/runs/%s/KerasANN/version%s/effs.txt' % (run,version),eff)
np.savetxt('/home/rosek1/ANN/runs/%s/KerasANN/version%s/thresholds.txt' % (run,version),thresh)

time=np.zeros(len(eval_gps_s))
for i in range(len(eval_gps_s)):
	time[i]=eval_gps_s[i]+eval_gps_ms[i]*.001
np.savetxt('/home/rosek1/ANN/runs/%s/KerasANN/version%s/time.txt' % (run,version),time)
np.savetxt('/home/rosek1/ANN/runs/%s/KerasANN/version%s/signif.txt' % (run,version),evalsignif) 

eff_err=sigma*np.sqrt((1-eff)*eff/numuncleaneval) #Error bar calculation
fap_err=sigma*np.sqrt((1-fap)*fap/numcleaneval)

np.save(eff_err_path_max,eff_err)
np.save(fap_err_path_max,fap_err)
