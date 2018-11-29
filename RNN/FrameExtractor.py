###########################################
# This script gathers raw input data for the Recurrent Neural Network.
# We use data from LIGO's auxiliary channels as our input to the RNN. 
# The RNN correlates activity in these auxiliary channels with glitches in
# LIGO's primary channel of information: the gravitational-wave channel.
# We use the package GWpy to access LIGO's data.  We whiten and downsample 
# the time series data with GWpy. This script also labels all times as either
# corresponding to a gravitational-wave channel glitch or not corresponding to
# a gravitational-wave glitch. These labels are used for training the RNN classifier. 
# 
# Author: Kyle Rose    6/23/18
##########################################
import sys
import os
import h5py
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from gwpy.segments import DataQualityFlag
from gwpy.segments import SegmentList
from laldetchar.idq import event
from laldetchar.idq import idq

print("Collecting and labelling Aux channel data")
run=31
version=11
gpsstart=1167536688 
dur=7*24*3600        
gpsstop=gpsstart+dur
samplerate=1024.
fftlength=32
overlap=8
flag='L1:DMT-ANALYSIS_READY:1'
gwchannel='L1_CAL-DELTAL_EXTERNAL_DQ_32_2048'
trigger_dir='/home/kyle.rose/idqtest_ann_with6thfeature/triggers'
signif_threshold=20
channellist='/home/kyle.rose/RNN/Bestchannels.txt' 
##############################################################################################################
### Define columns for later
col_kw = {
    'tstart': 0,
    'tstop': 1,
    'tcent': 2,
    'fcent': 3,
    'uenergy': 4,
    'nenergy': 5,
    'npix': 6,
    'signif': 7,
    }
col = col_kw.copy()

### Read in channels
with open(channellist) as f:
    channels = f.read().splitlines()
print(channels)

locktimes=DataQualityFlag.query_dqsegdb(flag,gpsstart,gpsstop)
print("Locktimes")
print(locktimes.active)
for i,seg in enumerate(locktimes.active): #Loop through lock segments
	print("Locksegment %d: %d-%d" % (i,seg[0],seg[1]))
	if (seg[1]-seg[0])<=fftlength: #Ignore lock segments that are too short 
		continue
	### Get triggers 
        print("Collecting triggers for labelling")
        trig_files = idq.get_all_files_in_range(trigger_dir, seg[0], seg[1], pad=0, suffix='.trg')  
        trigger_dict = event.loadkwm(trig_files)
        trigger_dict.include([[seg[0],seg[1]]])
        if trigger_dict[gwchannel]:
        	trigger_dict.apply_signif_threshold(threshold=signif_threshold,channels=[gwchannel])
        	darmtrg = trigger_dict.get_triggers_from_channel(gwchannel)
		auxdata=TimeSeriesDict.get(channels,seg[0],seg[1],frametype='L1_R',verbose=True) #Generate a dictionary for each
		for key, value in auxdata.iteritems():
			#print(value)
        		value.whiten(fftlength,overlap)  #Whiten the data
        		if value.sample_rate.value==samplerate:   #Convert all channels to the same samplingrate
                		continue
        		else:
				auxdata[key]=auxdata[key].resample(samplerate) 
				assert auxdata[key].sample_rate.value==1024. 
					
		datapath='/home/kyle.rose/RNN/runs/%d/version%d/Lockseg%d_%d-%d_Data' % (run,version,(i+10),seg[0],seg[1])	
		if not os.path.exists(os.path.dirname(datapath)):
        		try:
            			os.makedirs(os.path.dirname(datapath))
        		except OSError as exc: 
            			if exc.errno != errno.EEXIST:
                			raise
		print("Len(auxdata)")
		Length=len(auxdata['L1:ASC-REFL_A_RF9_I_PIT_OUT_DQ']) # Obtain length of timeseriesdict sequences, used for initializing labels array
		
		auxdata.write(datapath,format='hdf5') #Save each lock segment
		### Define time array
		time=np.arange(float(seg[0]),float(seg[1])+0/samplerate,(1/samplerate),dtype='Float128')
		#for el in time[0:1024*5]:
                        #print('%.9f'% el)
		labels=np.zeros(Length)
		assert(len(time)==len(labels))
		print("")
		print("Number of triggers in locksegment %d: %f" %(i,len(darmtrg)))	
		print("")
		for trig in darmtrg:
			print("trig")
			print(trig)
			startidx=np.where(time==int(trig[col['tstart']]))[0][0] #FIXME This might be bad if a glich happens Very soon after lock
			k=0
			while time[startidx]<trig[col['tstop']]:
				if time[startidx]>trig[col['tstart']]:
					labels[startidx]=1
					k=1						
				startidx+=1
				if startidx==len(labels):
					break
			if k==0:  ### If the glitch occured in between time steps, label the latter time step as glitch
				if startidx==len(labels):
					labels[startidx-1]=1
				else:
					labels[startidx]=1
	else:
		print("No gw channel glitches in locksegment %d" % (i))
		continue
 	timepath='/home/kyle.rose/RNN/runs/%d/version%d/Lockseg%d_%d-%d_GPStime' % (run,version,(i+10),seg[0],seg[1])
	labelpath='/home/kyle.rose/RNN/runs/%d/version%d/Lockseg%d_%d-%d_Labels' % (run,version,(i+10),seg[0],seg[1])
	np.savetxt(timepath,time)
	np.savetxt(labelpath,labels)
	numglitch=len(np.nonzero(labels)[0])	
	print("Glitches in locksegment %d: %d" % (i,numglitch))	

if len(locktimes.active)==0:
	print("No locktimes")
print("Done")
