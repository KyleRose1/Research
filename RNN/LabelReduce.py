################################################################
# This is one method for balancing the number of glitch/ 
# clean sequences given to an RNN.
# This script takes in raw aux channel data and a time series  
# of labels as output by  FrameExtractor.py. It generates a 
# sequence for each glitch in the label time series. It then randomly
# pulls as many clean times and generates sequences from them.
# All of the sequences of data are joined together into a single time series,
# and likewise for the sequences of labels. This can be passed to KerasDeepRNN.py
# where the timeseriesgenerator will separate the time series back into individual 
# sequences. Note: the stride for the timeseriesgenerator in KerasDeepRNN.py must 
# be equal to the length of the sequences.
#
# Author: Kyle Rose   6/4/18
#################################################################
import numpy as np
import math
import glob
import h5py

run=30
version=10
samplerate=1024  #Hz
window=1        #seconds
lengthofsequences=100
startindex=0
Framextractorrun=30 #Build data from FrameExtractor data in this ~/RNN/runs/__ directory

finallabelspath='/home/kyle.rose/RNN/runs/%s/vers%sprocessedlabels.npy' % (run, version)
finalsequencespath='/home/kyle.rose/RNN/runs/%s/vers%sprocessedsequences.npy' % (run, version)
finalgpstimepath='/home/kyle.rose/RNN/runs/%s/vers%sprocessedgpstimes.npy' % (run, version)
 
datafile_list = sorted(glob.glob('/home/kyle.rose/RNN/runs/%d/version%d' % (Framextractorrun,version) + '/*Data'))
labelfile_list= sorted(glob.glob('/home/kyle.rose/RNN/runs/%d/version%d' % (Framextractorrun,version) + '/*Labels'))
gpstimefile_list= sorted(glob.glob('/home/kyle.rose/RNN/runs/%d/version%d' % (Framextractorrun,version) + '/*GPStime'))
print(datafile_list)
print(labelfile_list)
print(gpstimefile_list)
assert(len(datafile_list)==len(labelfile_list))
assert(len(datafile_list)==len(gpstimefile_list))

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

lockarray={}
for i,filepath in enumerate(datafile_list):
	print("")
	print(filepath)
	print("")	
	hf = h5py.File(filepath, 'r') 
	p=0
	for key in list(hf.keys()):
		print(key)
		if p==0:
			snglchan=np.array([hf.get(key).value])
			data=snglchan
			print(len(data))
			p+=1
		else:	
			snglchan=np.array([hf.get(key).value])
			data = np.append(data,snglchan,axis=0) # `data` is now an ndarray.
			print("DATA")
			print(len(data))
			print(len(data[0]))
	data=data.T

	glitchindices=[]
	cleanindices=[]
	labels=np.loadtxt(labelfile_list[i])
	print("Labels")
	print(len(labels))
	gpstimes=np.loadtxt(gpstimefile_list[i])
	print("GPS times")
	print(len(gpstimes))
	assert(len(labels)==len(gpstimes))
	#Find the indices of the time/labels arrays where glitches occured
	print("Obtaining glitch times for lockstretch %d" % i)
	
	### For test where we only want one 1 for each glitch rather than multiple 1s.###
	### Remove all but the last 1 in each group of 1s   7/26/18 ###
	for p in range(len(labels)-1):
		if labels[p+1]==1:
			labels[p]=0
	####
	for v in range(len(labels)):
		if labels[v]==1:
			glitchindices.append(v) 

	print("Number of glitches for lockstretch %d" % i)
	print(len(glitchindices))

	#Randomly sample indices to get clean times, only keep those that aren't within a second of a glitch
	print("Obtaining Clean times")
	while len(cleanindices)<len(glitchindices):
		idx=np.random.randint(lengthofsequences,len(labels)) #Has to be enough room to make a sequence going backward from that index
		if np.abs(find_nearest(glitchindices,idx)-idx)>samplerate*window: #If the clean index is far enough away from a glitch index
			cleanindices.append(idx)                                  #Accept the clean index
		else:
			continue
	print("Number of cleans for lockstretch %d" % i)
	print(len(cleanindices))

	###Combine glitch and clean
	allindices=glitchindices+cleanindices
	allindices.sort()
	print("Length of indices array")
	print(len(allindices))
	pieces=[allindices]

	### Now create a 'Time series' of sequences and another for labels. These will be fed to the TimeSeriesGenerator for actual batch generation
	print("Generating Processed Time Series")
	k=0
	if len(allindices)==0:  ##This ensures that sequences and finallabels are redefined/not reused.. 
		continue    ## in this setup we never pull clean times from all clean lock stretches which probably isn't best
	else:
		for q,piece in enumerate(pieces):
                        print("Piece %d of %d" % (q,len(pieces)))
			for j,indx in enumerate(piece):
				print("Index %d of %d" % (j,len(piece)))
				if k==0:
					sequences=data[indx-lengthofsequences+1:indx+1,:] 
					finallabels=labels[indx-lengthofsequences+1:indx+1]
					finalgpstimes=gpstimes[indx-lengthofsequences+1:indx+1]
					k+=1
		
				else:
					labl=labels[indx-lengthofsequences+1:indx+1] #Create the sequence of labels
        				seq=data[indx-lengthofsequences+1:indx+1,:]  #Create the sequence
					gps=gpstimes[indx-lengthofsequences+1:indx+1] #Create the sequence of gps times
					finallabels=np.append(finallabels,labl)	     #Add them to the 'timeseries' of labels
					sequences=np.append(sequences,seq,axis=0)    #Add them to the 'timeseries' of sequences
					finalgpstimes=np.append(finalgpstimes,gps)   #Add them to the 'timeseries' of gpstimes	
					
			print("len(sequences)")
			print(len(sequences))
		
			if i==0:
				Trulyfinaldata=sequences
				Trulyfinallabels=finallabels
				Trulyfinalgps=finalgpstimes
				i=1
			else:
				print("SECONDTIME")
				Trulyfinaldata=np.append(Trulyfinaldata,sequences,axis=0)
				Trulyfinallabels=np.append(Trulyfinallabels,finallabels)
				Trulyfinalgps=np.append(Trulyfinalgps,finalgpstimes)	

assert(np.shape(Trulyfinallabels)==np.shape(Trulyfinalgps))
np.save(finallabelspath,Trulyfinallabels)
np.save(finalsequencespath,Trulyfinaldata)	
np.save(finalgpstimepath,Trulyfinalgps)

