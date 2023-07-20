import wfdb as wfdb
import matplotlib.pyplot as plt
import numpy as np


# https://blog.krybot.com/a?ID=01450-523f4738-dd72-4486-9c41-7911354d0a3a


# Draw ECG
def draw_ecg(x):
    plt.plot(x)
    plt.title('Raw_ECG')
    plt.show()
    
#Draw the ECG and its R wave position
def draw_ecg_R(record,annotation):
    plt.plot(record.p_signal) #Draw the ECG signal
    R_v=record.p_signal[annotation.sample] #Get R wave peak value
    plt.plot(annotation.sample,R_v,'or')#Draw R wave
    plt.title('Raw_ECG And R Position')
    plt.show()
def selData(record,annotation,label,R_left):
    a=annotation.symbol
    f=[k for k in range(len(a)) if a[k]==label] #Find the corresponding label R wave position index
    signal=record.p_signal
    R_pos=annotation.sample[f]
    res=[]
    for i in range(len(f)):
        if(R_pos[i]-R_left>0):
            res.append(signal[R_pos[i]-R_left:R_pos[i]-R_left+250])
    return res
        
# Read ECG data
def read_ecg_data(filePath,channel_names):
    '''
    Read ECG file
    sampfrom: Set the starting position for reading the ECG signal, sampfrom=0 means to start reading from 0, and the default starts from 0
    sampto: Set the end position of reading the ECG signal, sampto = 1500 means the end from 1500, the default is to read to the end of the file
    channel_names: set the name of reading ECG signal, it must be a list, channel_names=['MLII'] means reading MLII lead
    channels: Set the number of ECG signals to be read. It must be a list. Channels=[0, 3] means to read the 0th and 3rd signals. Note that the number of signals is uncertain 
    record = wfdb.rdrecord('../ecg_data/102', sampfrom=0, sampto = 1500) # read all channel signals
    record = wfdb.rdrecord('../ecg_data/203', sampfrom=0, sampto = 1500,channel_names=['MLII']) # Only read "MLII" signal
    record = wfdb.rdrecord('../ecg_data/101', sampfrom=0, sampto=3500, channels=[0]) # Only read the 0th signal (MLII)
    print(type(record)) # View record type
    print(dir(record)) # View methods and attributes in the class
    print(record.p_signal) # Obtain the ECG lead signal, this article obtains MLII and V1 signal data
    print(record.n_sig) # View the number of lead lines
    print(record.sig_name) # View the signal name (list), the lead name of this text ['MLII','V1']
    print(record.fs) # View the adoption rate
    '''
    
    record = wfdb.rdrecord(filePath,channel_names=[channel_names])
    print('Number of lead lines:')
    print(record.n_sig) # View the number of lead lines
    print('Signal name (list)')
    print(record.sig_name) # View the signal name (list), the lead name of this text ['MLII','V1']

    '''
    Read annotation file
    sampfrom: Set the starting position for reading the ECG signal, sampfrom=0 means to start reading from 0, and the default starts from 0
    sampto: Set the end position of reading the ECG signal, sampto = 1500 means the end from 1500, the default is to read to the end of the file
    print(type(annotation)) # View the annotation type
    print(dir(annotation))# View methods and attributes in the class
    print(annotation.sample) # Mark the sharp position of the R wave of each heartbeat, corresponding to the ECG signal
    annotation.symbol #Mark the type of each heartbeat N, L, R, etc.
    print(annotation.ann_len) # The number of labels
    print(annotation.record_name) # The file name to be marked
    print(wfdb.show_ann_labels()) # View the type of heartbeat
    '''
    annotation = wfdb.rdann(filePath,'atr')
# print(annotation.symbol)
    return record,annotation

if __name__ == "__main__":
    filePath='G:\MLDataSets.2022\Physionet-Non-EEG Dataset for Assessment of Neurological Status/Subject1_AccTempEDA'
    Channel_Name='EDA'
    record,annotation=read_ecg_data(filePath,Channel_Name)
# draw_ecg(record.p_signal)

    draw_ecg_R(record,annotation)
    res=selData(record,annotation,'N',100)
    print(len(res))
    plt.plot(res[20])
    
    plt.show()