import threading
import time
import pandas as pd
from io import StringIO
import sklearn
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import xml.etree.ElementTree as ET
import os
import ecg_plot
import matplotlib.pyplot as plt
import csv
import seaborn as sns

# create config for constants, filepaths
# push code to github
# ask professor if i need all 12 wqves to make predictions or maybe the first 6 are most important and distinguishable 
# maybe i could do undersampling on the whole dataset since no dublicates will be added
# check 0 and 1 inconfusion matrix, check recall and precision, save as png
# would be nice to see a ration of majority class to minority class
# plot noraml and abnormal ecgs on the same plot
# use only one lead! use spectograms
# cross validation for cnn is possible, do it
# constany seed!
# use gpus to train models?
# for binary classification i combined the recoed with normal and otherwise normal diagnoses, it decreased percentage of false negatives from 21% to 16%
# normally 703 files get skipped, adding bmi check changed it to 
# sample rate - 500Hz!

def raw_wave_processing(wave):
    wave = re.sub(r"\s+", "", wave)
    return list([int(num) for num in wave.split(',')])

def get_waveform(df, column='WaveformData'):
    return [raw_wave_processing(wave) for wave in df[column]]

def get_1lead_waveform(df, column='WaveformData'):
    return [raw_wave_processing(df[column][0])]

def get_1lead_normalBMI_waveforms(df, column='WaveformData'):
    return [raw_wave_processing(df[column][0])]

# TODO: should also use DTW?
def normalize_waveforms(waves):
    scaler = MinMaxScaler((-1, 1))
    return scaler.fit_transform(waves)

def get_raw_diagnosis(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    diagnosis = ''
    for inter in root.findall('Interpretation'):
        obj = inter.find('Diagnosis')
        if obj != None:
            for diag in obj:
                diagnosis += diag.text + '#'
    return diagnosis[:-1]

def is_normal_BMI(df):
    h_cm = df['Height'].values[0]/100
    w = df['Weight'].values[0]
    bmi = w/(h_cm*h_cm)
    return bmi < 18.5 or bmi > 25.0
    
def read_file(file_path, read_waveform_function, check_bmi=False):
    # contains the waveform data of the ecg strip for the given lead
    xpath = '//RestingECGMeasurements/MedianSamples/WaveformData'
    xpath_patientInfo = '//PatientInfo'
    normal_label1 = '#Normal ECG#'
    normal_label2 = '#Otherwise normal ECG#'
    try:
        df = pd.read_xml(file_path, xpath=xpath)
        if check_bmi:
            df_patientInfo = pd.read_xml(file_path, xpath=xpath_patientInfo)
    except ValueError as value_error:
        print('file_name', file_path, 'error', value_error)
        return ([], 0, 0)
    except Exception as e:
        print('file_name', file_path, 'error', e)
        return ([], 0, 0)

    # remove underweight, overweight and obese patients (for now)
    if check_bmi and is_normal_BMI(df_patientInfo):
        return ([], 0, 0)
        
    waves = read_waveform_function(df)
    res_waves = np.array(waves).ravel()

    if len(res_waves) == 0:
        return (res_waves, 0, 0)

    diagnosis = get_raw_diagnosis(file_path)
    reversed_diagnosis = diagnosis[::-1]
    position = reversed_diagnosis.find('#')
    new_diagnosis = reversed_diagnosis[position+1:][::-1]
    
    # diagnosis might be empty (no diagnosis provided), should skip such entries!
    if len(new_diagnosis) == 0 or 'no ECG analysis possible' in new_diagnosis:
        return ([], 0, 0)

    label_2_classes = 1
    if normal_label1 in new_diagnosis or normal_label2 in new_diagnosis:
        label_2_classes = 0
        
    label_3_classes = 0
    if '#Normal ECG#' in new_diagnosis or '#Otherwise normal ECG#' in new_diagnosis:
        label_3_classes = 0
    elif '#Abnormal ECG#' in new_diagnosis:
        label_3_classes = 1
    elif '#Borderline ECG#' in new_diagnosis:
        label_3_classes = 2
    else:
        return ([], 0, 0)
    return (res_waves, label_2_classes, label_3_classes)
        

def multi_threaded_file_reader(file_paths, read_waveform_function, check_bmi):
    threads = []
    results = {}

    def read_file_thread(file_path):
        result = read_file(file_path, read_waveform_function, check_bmi=check_bmi)
        results[file_path] = result

    for file_path in file_paths:
        thread = threading.Thread(target=read_file_thread, args=(file_path,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results

def file_names(dirs):
    results = []
    for dir in dirs:
        for file in os.listdir(dir):
            results.append(os.path.join(dir, file))
    return np.array(results)

def write_into_csv(data, filename):
    with open(filename, 'w') as f:
        mywriter = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC, delimiter=',')
        mywriter.writerows(data)

# persist data to access it faster next time
def persist_data(waves, labels2, labels3, waves_file, labels2_file, labels3_file):
    write_into_csv(waves, waves_file)
    write_into_csv(labels2, labels2_file)
    write_into_csv(labels3, labels3_file)

def read_data(dirs, read_waveform_function, check_bmi):
    X, data_labels_2_classes, data_labels_3_classes, skipped_files = [], [],[], 0
    file_paths = file_names(dirs)
    st = time.time()
    results = multi_threaded_file_reader(file_paths, read_waveform_function, check_bmi)
    for _, content in results.items():
        if len(content[0]) == 0:
            skipped_files += 1
            continue
        else:
            X.append(content[0])
            data_labels_2_classes.append(content[1])
            data_labels_3_classes.append(content[2])
    X = np.array(normalize_waveforms(X))
    data_labels_2_classes = np.array(data_labels_2_classes)
    data_labels_3_classes = np.array(data_labels_3_classes)
    print("shape X = ", X.shape, "shape y = ", data_labels_2_classes.shape, "skipped", skipped_files, "files/entries")
    
    elapsed_time = time.time() - st
    print('Reading waves time (full):', elapsed_time/60, 'minutes')
    return X, data_labels_2_classes, data_labels_3_classes

def read_data_from_file(filename):
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quoting = csv.QUOTE_NONNUMERIC)
        data = [row for row in spamreader]
    return np.array(data)

# split into training and testing datasets
def split_dataset(data, labels, training_percent=0.75):
    training_proportion = int(training_percent * len(data))

    training_X = data[:training_proportion]
    testing_X = data[training_proportion:]
    training_y = labels[:training_proportion]
    testing_y = labels[training_proportion:]

    return training_X, testing_X, training_y, testing_y

def undersample(X, y):
    cc = ClusterCentroids(
        estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42, sampling_strategy='not minority'
    )
    st = time.time()
    X_resampled, y_resampled = cc.fit_resample(X, y)
    elapsed_time = time.time() - st
    print('Undersampling time:', elapsed_time/60, 'minutes')
    
    print('Original dataset shape:', Counter(y))
    print('Resampled dataset shape:', Counter(y_resampled))
    return X_resampled, y_resampled

def depict_confusion_matrix(cf_matrix, title):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.title(title)
    plt.show()

def reverse_one_hot(predictions):
    reversed_x = []
    for x in predictions:
        reversed_x.append(np.argmax(np.array(x)))
    return reversed_x

def learn_the_model(training_set_X, training_set_y, testing_set_X, testing_set_y, image_shape = (24, 300)):

    test_labels = tf.keras.utils.to_categorical(testing_set_y, 2)
    train_labels = tf.keras.utils.to_categorical(training_set_y, 2)
    
    train_images = training_set_X.reshape(training_set_X.shape[0], image_shape[0], image_shape[1], 1)
    test_images = testing_set_X.reshape(testing_set_X.shape[0], image_shape[0], image_shape[1], 1)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_shape[0], image_shape[1], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(), metrics=['accuracy', 'mae', 'mse'])
    
    train_data_size = train_images.shape[0]
    test_data_size = test_images.shape[0]
    
    print("model will be trained with {} and be tested with {} sample".format(train_data_size, test_data_size))
    print("Fitting model to the training data...")
    model.fit(train_images, train_labels, batch_size=150, epochs=5, verbose=1, validation_data=None)
    
    predictions_test = model.predict(test_images, batch_size=150, verbose=1)
    predictions_train = model.predict(train_images, batch_size=150, verbose=1)
    print(model.evaluate(test_images, test_labels, batch_size=150, verbose=1))
    return predictions_test, predictions_train

def train_and_test_CNN(X_resampled, y_resampled, testing_X, testing_y):
    predictions_CC_test, predictions_CC_train = learn_the_model(X_resampled, y_resampled, testing_X, testing_y)
    print("Evaluation accuracy score (test) = ", accuracy_score(testing_y, reverse_one_hot(predictions_CC_test)))
    print("Evaluation accuracy score (train) = ", accuracy_score(y_resampled, reverse_one_hot(predictions_CC_train)))
    
    cf_matrix = confusion_matrix(testing_y, reverse_one_hot(predictions_CC_test))
    depict_confusion_matrix(cf_matrix, 'Testing set (full)')
    
    cf_matrix = confusion_matrix(y_resampled, reverse_one_hot(predictions_CC_train))
    depict_confusion_matrix(cf_matrix, 'Training set (full)')
    
    
if __name__ == "__main__":
    dirs = ['/groups/umcg-endocrinology/tmp02/projects/ukb-55495/data/metaData/v5/xml_T3/', "/groups/umcg-endocrinology/tmp02/projects/ukb-55495/data/metaData/v5/xml_T2/"]
    # read and persist only the first lead
    # X, data_labels_2_classes, data_labels_3_classes = read_data(dirs, get_1lead_waveform)
    # read and persist all the 12 leads
    # X, data_labels_2_classes, data_labels_3_classes = read_data(dirs, get_waveform)
    # read and persist only the first lead for the patients within the normal bmi range
    # the same amount of files was skipped wihch means that all the patients are within the normal bmi range
    X, data_labels_2_classes, data_labels_3_classes = read_data(dirs, get_1lead_waveform, True)

    persist_data(X, [data_labels_2_classes], [data_labels_3_classes], 'waves_full_1lead_normalBMI.csv', 'labels_full_2_classes_1lead_normalBMI.csv', 'labels_full_3_classes_1lead_normalBMI.csv')
    
    # st = time.time()
    # waves = read_data_from_file('waves_full.csv')
    # print(waves.shape)
    
    # labels2 = read_data_from_file('labels_full_2_classes.csv')[0]
    # print(labels2.shape)

    # labels3 = read_data_from_file('labels_full_3_classes.csv')[0]
    # print(labels3.shape)
    # elapsed_time = time.time() - st
    # print('Reading data time:', elapsed_time/60, 'minutes')

    # training_X, testing_X, training_y, testing_y = split_dataset(waves, labels2, 0.75)
    # X_resampled, y_resampled = undersample(training_X, training_y)
    
    # st = time.time()
    # train_and_test_CNN(X_resampled, y_resampled, testing_X, testing_y)
    # elapsed_time = time.time() - st
    # print('Training data time:', elapsed_time/60, 'minutes')
    



