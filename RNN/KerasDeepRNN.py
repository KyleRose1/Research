########################################################
# A Recurrent Neural Network
#
# This script takes in preprocessed time series of 
# training and evaluation data.  It generates batches
# of sequences from that data and then trains and 
# evaluates the RNN
#
# Author: Kyle Rose 6/3/18
########################################################
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Flatten
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import Callback , EarlyStopping, ModelCheckpoint
from sklearn import metrics
from keras.models import load_model
import numpy as np
import sys
import os
import glob
run=31
version=3
sigma=3      

#model_path_max = '/home/rosek/RNN/runs/%s/version%s/model_max_%s' % (run,version,version)
#model_path_end = '/home/rosek/RNN/runs/%s/version%s/model_end_%s' % (run,version,version)
eff_path_max = '/home/rosek1/RNN/runs/%s/version%s/eff_maxrun%svers%s' % (run,version,run,version)
fap_path_max =   '/home/rosek1/RNN/runs/%s/version%s/fap_maxrun%svers%s' % (run,version,run,version)
eff_err_path_max='/home/rosek1/RNN/runs/%s/version%s/eff_err_maxrun%svers%s' % (run,version,run,version)
fap_err_path_max='/home/rosek1/RNN/runs/%s/version%s/fap_err_maxrun%svers%s' % (run,version,run,version)
threshpath='/home/rosek1/RNN/runs/%s/version%s/thresholdsrun%svers%s' % (run,version,run,version)
labelspath='/home/rosek1/RNN/runs/%s/version%s/labelsrun%svers%s' % (run,version,run,version)
predictionspath='/home/rosek1/RNN/runs/%s/version%s/predictions_run%s_vers%s' % (run,version,run,version)
bestlossmodelpath='/home/rosek1/RNN/runs/%s/version%s/bestlossmodel_run%svers%s.h5' % (run,version,run,version)
bestvallossmodelpath='/home/rosek1/RNN/runs/%s/version%s/bestvalidlossmodel_run%svers%s.h5' %(run,version,run,version)
evalgpstimespath='/home/rosek1/RNN/runs/%s/version%s/evalgpstimes_run%s_vers%s' % (run,version,run,version)
#Make Directories if they don't exist
paths=[fap_path_max,eff_path_max,eff_err_path_max,fap_err_path_max,labelspath,predictionspath,bestlossmodelpath,bestvallossmodelpath]
for path in paths:
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
#sys.stdout.close()
#print(sys.stdout)
#sys.stdout = sys.__stdout__
#print(sys.stdout)
###Redirect Standard output to a file 
#orig_stdout = sys.stdout    #Do this so we can reset the stdout at the end
#f = open('/home/rosek1/RNN/runs/%s/version%s/RNNrun%svers%sinfo.out' % (run,version,run,version),'w')
#sys.stdout=f
#print(sys.stdout)
#Batchsize=number of sequences in a batch
#Timesteps=The length of our sequences (Assuming no sample rate adjustment!)
#numauxchans=The number of aux channels we're using
#percenttrain is the percentage of the data used for training. (The rest is used for evaluation)
batchsize=128
timesteps=100
numauxchans=37
percenttrain=.95

#Load Aux channel data/ labels/
print("Loading Data")
data_list1  = sorted(glob.glob('/home/rosek1/RNN/runs/%d' % (run) + '/*sequences.npy'))
labels_list1= sorted(glob.glob('/home/rosek1/RNN/runs/%d' % (run) + '/*labels.npy'))
gps_list1   = sorted(glob.glob('/home/rosek1/RNN/runs/%d' % (run) + '/*gpstimes.npy'))
assert(len(data_list1)==len(labels_list1))
assert(len(data_list1)==len(gps_list1))
print("datalist")
print(data_list1)
print("labellist")
print(labels_list1)
print("gps_list")
print(gps_list1)
### glob doesn't always give the files in the right order, so order them correctly here:
#sorteddata_list=[]
#sortedlabels_list=[]

#for elm in data_list:
#        print(int(elm[29:31]))
#	sorteddata_list.append(int(elm[29:31]))
#sorteddata_list.sort()
#data_list1=[]
#for elem in sorteddata_list:
#	data_list1.append('/home/rosek1/RNN/runs/%d' % (run) + '/vers'  + '%d' % (elem) + 'processedsequences.npy')	
#print(data_list1) 
#
#for elm in labels_list:
#	print(int(elm[29:31]))
#	sortedlabels_list.append(int(elm[29:31]))
#sortedlabels_list.sort()
#labels_list1=[]
#for elem in sortedlabels_list:
#	labels_list1.append('/home/rosek1/RNN/runs/%d' % (run) + '/vers' + '%d' % (elem) + 'processedlabels.npy')
#print(labels_list1)

### Load in all of the data and make 1 huge array of sequences, 1 of labels, and 1 of gps times
for i in range(len(data_list1)):
	if i==0:
		print(data_list1[i])
		alldata=np.load(data_list1[i])
		alllabels=np.load(labels_list1[i])
		allgps=np.load(gps_list1[i])
		print(len(alldata))
		print(np.shape(alldata))
		print(len(alllabels))
		if len(alldata)!=len(alllabels) or len(alldata)!=len(allgps):
			print("DATA and LABELS or GPS NOT EQUAL")
	else:
                print(data_list1[i])
		tempdata =np.load(data_list1[i])
		templabel=np.load(labels_list1[i])
		tempgps  =np.load(gps_list1[i])
		alldata  =np.append(alldata,tempdata,axis=0)
		alllabels=np.append(alllabels,templabel)
		allgps   =np.append(allgps,tempgps)	
		print(np.shape(alldata))
                print(len(alllabels))
		if len(tempdata)!=len(templabel) or len(tempdata)!=len(tempgps):
                           print("DATA and LABELS or GPS  NOT EQUAL ")

print("All of the data:")
print(np.shape(alldata))
print(np.shape(alllabels))
print(np.shape(allgps))
assert(np.shape(allgps)==np.shape(alllabels))

### The length of data/labels should be an integer multiple of the number of timesteps
assert(len(alllabels)%timesteps==0) 

### Divide into training/evaluation
numofseqindata=len(alllabels)/timesteps
cutoff=int(percenttrain*numofseqindata)
cutoffindex=timesteps*cutoff

traindata  =alldata[0:cutoffindex,:]
trainlabels=alllabels[0:cutoffindex]
traingps   =allgps[0:cutoffindex]

evaldata  =alldata[cutoffindex:,:]
evallabels=alllabels[cutoffindex:]
evalgps   =allgps[cutoffindex:]

print("Training data:")
print(np.shape(traindata))
print("Eval data:")
print(np.shape(evaldata))

assert(len(traindata)%timesteps==0)
assert(len(trainlabels)%timesteps==0)
assert(len(traingps)%timesteps==0)
assert(len(evaldata)%timesteps==0)
assert(len(evallabels)%timesteps==0)
assert(len(traindata)==len(trainlabels))
assert(len(evaldata)==len(evallabels))

#traindata=np.load("/home/rosek/RNN/runs/5/trainprocessedsequences.npy")
#trainlabels=np.load("/home/rosek/RNN/runs/5/trainprocessedlabels.npy")
#evaldata=np.load("/home/rosek/RNN/runs/5/evalprocessedsequences.npy")
#evallabels=np.load("/home/rosek/RNN/runs/5/evalprocessedlabels.npy")

#Offset the labels and gps times by one digit to compensate for the timeseries generator
trainlabels=np.insert(trainlabels,0,0)    
evallabels =np.insert(evallabels,0,0)
traingps   =np.insert(traingps,0,0)
evalgps    =np.insert(evalgps,0,0)

print("After insertion of 0 to the label arrays:")
print(np.shape(traindata))
print(np.shape(trainlabels))
print(np.shape(evaldata))
print(np.shape(evallabels))
assert(np.shape(traingps)==np.shape(trainlabels))
assert(np.shape(evalgps)==np.shape(evallabels))
#Make the data 2-d if not already:
if traindata.ndim==1:
	traindata=np.reshape(traindata,(len(traindata),1))
if evaldata.ndim==1:
        evaldata=np.reshape(evaldata,(len(evaldata),1))
if trainlabels.ndim==1:
        trainlabels=np.reshape(trainlabels,(len(trainlabels),1))
if evallabels.ndim==1:
        evallabels=np.reshape(evallabels,(len(evallabels),1))
if traingps.ndim==1:
	traingps=np.reshape(traingps,(len(trainlabels),1))
if evalgps.ndim==1:
	evalgps=np.reshape(evalgps,(len(evalgps),1))
print("After making 2d if not already:")
print(np.shape(traindata))
print(np.shape(trainlabels))
print(np.shape(evaldata))
print(np.shape(evallabels))
print(np.shape(evalgps))
assert(np.shape(evalgps)==np.shape(evallabels))
### Trim the timeseries so that they play nicely with the timeseries generator
### This is a bit weird. We keep one sequence of data more than goes evenly into the batches to offset
### the fact that the labels are 'one ahead'  Note this last sequence of data is Not used for training,
### it is just there so that we maximize the number of full batches without creating any small batches
### that trip up the RNN.  
if len(traindata)%(batchsize*timesteps)==0:
	traincut=batchsize*timesteps-1
else:
	traincut=len(traindata)%(batchsize*timesteps)-1 
traindata  =traindata[:-traincut,:]
trainlabels=trainlabels[:-traincut-1]
traingps   =traingps[:-traincut-1]

if len(evaldata)%(batchsize*timesteps)==0:
        evalcut=batchsize*timesteps-1
else:
	print("len(evaldata)")
	print(len(evaldata))
	print("batchsize")
	print(batchsize)
	print("timesteps")
	print(timesteps)
	print("len(evaldata)%(batchsize*timesteps)-1")
        evalcut=len(evaldata)%(batchsize*timesteps)-1
	print(evalcut)
print("evalcut")
print(evalcut)
print("np.shape(evaldata)")
print(np.shape(evaldata))
evaldata  =evaldata[:-evalcut,:]
print("np.shape(evaldata[:-evalcut,:])")
print(np.shape(evaldata))
evallabels=evallabels[:-evalcut-1]
evalgps   =evalgps[:-evalcut-1]

assert(len(traindata)==len(trainlabels))
assert(len(evaldata)==len(evallabels))
assert(len(evallabels)==len(evalgps))
assert(len(traindata)%timesteps==1)
assert(len(trainlabels)%timesteps==1)
assert(len(evaldata)%timesteps==1)
assert(len(evallabels)%timesteps==1)
assert(len(evaldata)%(batchsize*timesteps)==1)
assert(len(traindata)%(batchsize*timesteps)==1)
assert(len(traingps)%(batchsize*timesteps)==1)

numbatcheseval=int(len(evaldata)/(batchsize*timesteps))
numbatchestrain=int(len(traindata)/(batchsize*timesteps))

#Calculate number of cleans/uncleans in evaluation set for error bars later
numuncleantrain=0
numcleantrain=0
numuncleaneval=0
numcleaneval=0

numtrainsequences=int((len(traindata)-1)/timesteps)
for i in range(numtrainsequences):  #Don't take into account the last sequence, because it isn't used as data
	truetrainlab=trainlabels[1:]  
	if truetrainlab[timesteps*(i+1)-1]==1:
		numuncleantrain+=1
	elif truetrainlab[timesteps*(i+1)-1]==0:
		numcleantrain+=1
	else:
		print("something's wrong")
numevalsequences=int((len(evaldata)-1)/timesteps)
print("Len(evaldata)")
print(len(evaldata))
print("timesteps")
print(timesteps)
for i in range(numevalsequences):
        trueevallab=evallabels[1:]
        if trueevallab[timesteps*(i+1)-1]==1:
                numuncleaneval+=1
        elif trueevallab[timesteps*(i+1)-1]==0:
                numcleaneval+=1
        else:
                print("something's wrong")
assert(numcleaneval+numuncleaneval==numevalsequences)
assert(numcleantrain+numuncleantrain==numtrainsequences)
print("Number of training sequences: %d" % numtrainsequences)
print("Number of glitch sequences: %d" % numuncleantrain)
print("Number of clean sequences: %d" % numcleantrain)
print("")
print("Number of evaluation sequences: %d" % numevalsequences)
print("Number of glitch sequences: %d" % numuncleaneval)
print("Number of clean sequences: %d" % numcleaneval)

#Produce Training and Evaluation Batches
print("Produce Training Batches")
data_gen=TimeseriesGenerator(traindata,trainlabels,length=timesteps,\
sampling_rate=1,stride=timesteps,batch_size=batchsize,start_index=0)
print("Produce Evaluation Batches")
data_gen_eval=TimeseriesGenerator(evaldata,evallabels,
                                   length=timesteps,sampling_rate=1,
                                   stride=timesteps,batch_size=batchsize,start_index=0)

GPS_gen_eval=TimeseriesGenerator(evalgps,evalgps,length=timesteps,sampling_rate=1,stride=timesteps,batch_size=batchsize,start_index=0)

###Quick check of 'cut' logic above.
assert(len(data_gen[numbatchestrain-1][0])==batchsize)
assert(len(data_gen_eval[numbatcheseval-1][0])==batchsize)
assert(len(data_gen[numbatchestrain][0])==0)
assert(len(data_gen_eval[numbatcheseval][0])==0)
assert(len(GPS_gen_eval[numbatcheseval-1][0])==batchsize)
assert(len(GPS_gen_eval[numbatcheseval][0])==0)
### Define Model Hyperparams
print("Defining Model")
lstmsize=100
activation='tanh'
recurrent_activation='hard_sigmoid'
return_sequences=True
stateful=False
### Final layer:
densesize=1
denseactivation='sigmoid'
### Compile
loss='binary_crossentropy'
optimizer=keras.optimizers.Adam(lr=.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#'adam'
metrics1=['accuracy']
### Callbacks/Earlystopping
erlystpmonitor='val_loss'
bestlossmonitor='loss'
bestvallossmonitor='val_loss'
min_delta=.0001
patience=50
mode='auto'
#Train
trainingdata=data_gen
epochs=30000
validationdata=data_gen_eval
shuffle=True
#Evaluate
evaluationdata=data_gen_eval

model = load_model('/home/rosek1/RNN/runs/31/version1/bestvalidlossmodel_run31vers1.h5')

################ Actually Define Model ######################3
#model=Sequential()
#model.add(LSTM(lstmsize,batch_size=batchsize, input_shape=(timesteps,numauxchans),\
#	activation=activation,recurrent_activation=recurrent_activation,\
#	return_sequences=return_sequences,stateful=stateful,unit_forget_bias=True))
#model.add(LSTM(100,activation='tanh',recurrent_activation='hard_sigmoid',return_sequences=False,\
#	stateful=False,unit_forget_bias=True))
#model.add(LSTM(100,activation='tanh',recurrent_activation='hard_sigmoid',return_sequences=False,\
#	stateful=False))
#model.add(Dense(densesize,activation=denseactivation))
model.compile(loss=loss, optimizer=optimizer, metrics=metrics1)

### Print for future reference
print("model=Sequential()")
#print("model.add(LSTM(%s,batch_size=%s, input_shape=(%s,%s),activation=%s,recurrent_activation=%s,return_sequences=%s,stateful=%s))"% (lstmsize,batchsize,timesteps,numauxchans,activation,recurrent_activation,return_sequences,stateful))
#print('model.add(LSTM(4000,activation=softsign,recurrent_activation=hard_sigmoid,return_sequences=True,stateful=False)')
#print('model.add(LSTM(1000,activation=tanh,recurrent_activation=hard_sigmoid,return_sequences=False,stateful=False)')
##print("model.add(Flatten())")
#print("model.add(Dense(%s,activation=%s))"%(densesize,denseactivation))
#print("model.compile(loss=%s, optimizer=%s, metrics=%s)"%(loss,optimizer,metrics1))
print(model.summary())

callbacks1=[EarlyStopping(monitor=erlystpmonitor,min_delta=min_delta,patience=patience,mode=mode),ModelCheckpoint(bestlossmodelpath,monitor=bestlossmonitor,verbose=0,save_best_only=True),ModelCheckpoint(bestvallossmodelpath,monitor=bestvallossmonitor,verbose=0,save_best_only=True)]
print("callbacks1=[EarlyStopping(monitor=%s,min_delta=%s,patience=%s,mode=%s),ModelCheckpoint(bestlossmodelpath,monitor=%s,verbose=0,save_best_only=True),ModelCheckpoint(bestvallossmodelpath,monitor=bestvallossmonitor,verbose=0,save_best_only=True)]"% (erlystpmonitor,min_delta,patience,mode,bestlossmonitor))
print(" ")

print("Begin Training")
#model.fit_generator(trainingdata,epochs=epochs,verbose=2,callbacks=callbacks1,validation_data=validationdata,shuffle=shuffle)
print("model.fit_generator(%s,epochs=%s,verbose=2,callbacks=callbacks1,validation_data=%s,shuffle=%s)"%(trainingdata,epochs,validationdata,shuffle))
print("Done Training")

#Save the model
#print("Saving Model")
#model.save(modelpath)

#print(model.metrics_names)
#model.evaluate_generator(evaluationdata)
#print('model.evaluate_generator(%s)'% (evaluationdata))



#model = load_model('/home/rosek/RNN/runs/5/version7/model_run5vers7.h5')


print("Begin Evaluation")
predictions=model.predict_generator(evaluationdata,verbose=1)
print('predictions=model.predict_generator(%s,verbose=1)'% (evaluationdata))
print('predictions')
print(predictions)

print("Done Evaluation")

evalgpstimes=GPS_gen_eval

#Format the labels nicely
k=0
print(len(evaluationdata))
for i in range(len(evaluationdata)):
        if k==0:
                lab=evaluationdata[i][1]
                gps=evalgpstimes[i][1]
		labls=np.reshape(lab,(len(lab),))
		gps2=np.reshape(gps,(len(gps),))
                assert(len(lab)==len(gps))
		k+=1
        else:
                lab=evaluationdata[i][1]
		gps=evalgpstimes[i][1]
                lab=np.reshape(lab,(len(lab),))
		gps=np.reshape(gps,(len(gps),))
                labls=np.append(labls,lab)
		gps2=np.append(gps2,gps)

assert(len(labls)==len(gps2))
#Get ROC curve info
fap, eff, thresh=metrics.roc_curve(labls,predictions)
auc=metrics.auc(fap,eff)

np.save(threshpath,thresh)
np.save(labelspath,labls)
np.save(predictionspath,predictions)
np.save(evalgpstimespath,gps2)
np.save(fap_path_max,fap)
np.save(eff_path_max,eff)
#np.save(auc_path_max,auc)

eff_err=sigma*np.sqrt((1-eff)*eff/numuncleaneval) #Error bar calculation
fap_err=sigma*np.sqrt((1-fap)*fap/numcleaneval)

np.save(eff_err_path_max,eff_err)
np.save(fap_err_path_max,fap_err)
#Reset standard output and close the file.
#sys.stdout=orig_stdout
#f.close()

