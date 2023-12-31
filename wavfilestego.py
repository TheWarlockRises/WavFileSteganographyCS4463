"""

Programmed by Joseph D'Amico and Christopher Torres

An implementation of Detecting information-hiding in WAV audios 
by Qingzhong Liu, Andrew H. Sung, and Mengyu Qiao
https://ieeexplore.ieee.org/document/4761650

"""

import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
import sys
import csv

choice = 0
lsbBitNum = 0
testing_files, testing_labels = "testing_data_folders", "testing_labels_folder"
training_files, training_labels = "training_data_folders", "training_labels_folder"
steg_files = "steganalysis_files"

def extract_features(audio):
    """
    Extract joint distribution and Markov features from 
    second order derivative of audio signal
    """
    
    # Take second order derivative
    derivative = savgol_filter(audio, window_length=5, polyorder=2, deriv=2)

    # Quantize to [-8, 8]
    min_val = -8
    max_val = 8
    derivative = np.round((derivative - np.min(derivative)) / (np.max(derivative) - np.min(derivative)) * (max_val - min_val)) + min_val
    derivative = derivative.astype(int)
    
    # Build joint and conditional matrices
    joint_matrix = np.zeros((17,17))
    cond_matrix = np.zeros((17,17))
    N = len(derivative)
    for i in range(-8,9):
        for j in range(-8,9):
            joint_matrix[i+8,j+8] = np.sum((derivative[1:N-2] == i) & (derivative[2:N-1] == j)) / (N-3)
            cond_count = np.sum(derivative[2:-2] == i)
            if cond_count > 0 or cond_count < 0:
                cond_matrix[i+8,j+8] = np.sum((derivative[2:-2] == i) & (derivative[3:-1] == j)) / cond_count
            else:
                cond_matrix[i+8,j+8] = 0
    
    # Randomly flip LSB and extract features from modified audio
    bit_flips = np.random.choice([0,1], size=len(audio))
    modified = audio.copy()
    modified[bit_flips == 1] = -modified[bit_flips == 1]
    #modified[np.random.randn(len(audio)) > 0] = modified[np.random.randn(len(audio)) > 0] ^ 1
    
    mod_deriv = savgol_filter(modified, window_length=5, polyorder=2, deriv=2)

    # Rescale derivative values to range from -1 to 1
    mod_deriv = 2 * (mod_deriv - np.min(mod_deriv)) / (np.max(mod_deriv) - np.min(mod_deriv)) - 1

    # Quantize derivative values to integers from -8 to 8
    mod_deriv = np.round(mod_deriv * 8).astype(int)

    mod_joint = np.zeros((17,17))
    mod_cond = np.zeros((17,17))
    for i in range(-8,9):
        for j in range(-8,9):
            mod_joint[i+8,j+8] = np.sum((mod_deriv[2:-2] == i) & (mod_deriv[3:-1] == j))
            mod_cond_count = np.sum(mod_deriv[2:-2] == i)
            if mod_cond_count > 0 or mod_cond_count < 0:
                mod_cond[i+8,j+8] = np.sum((mod_deriv[2:-2] == i) & (mod_deriv[3:-1] == j)) / np.sum(mod_deriv[2:-2] == i)
            else:
                mod_cond[i+8,j+8] = 0

    # Calculate difference        
    joint_diff = joint_matrix - mod_joint
    cond_diff = cond_matrix - mod_cond
    
    # Return all 4 feature sets

    return joint_matrix, cond_matrix, joint_diff, cond_diff

def select_features(features):
    """
    Select relevant features using method described in paper
    """
    selected = []
    
    joint, cond, joint_diff, cond_diff = features

    #Threshold filters out features with low occurrence while keeping more common ones
    threshold = np.percentile(joint, 80)
    
    for i in range(17):
        for j in range(17):
            if np.abs(joint[i,j]) > threshold:
                selected.append(joint[i,j])
                selected.append(cond[i,j])
                selected.append(joint_diff[i,j])
                selected.append(cond_diff[i,j])
                
            if np.abs(cond[i,j]) > threshold:
                selected.append(joint[i,j])
                selected.append(cond[i,j])
                selected.append(joint_diff[i,j])
                selected.append(cond_diff[i,j])
                
            if np.abs(joint_diff[i,j]) > threshold:
                selected.append(joint[i,j])
                selected.append(cond[i,j])
                selected.append(joint_diff[i,j])
                selected.append(cond_diff[i,j])
                
            if np.abs(cond_diff[i,j]) > threshold:
                selected.append(joint[i,j])
                selected.append(cond[i,j])
                selected.append(joint_diff[i,j])
                selected.append(cond_diff[i,j])
     
    return np.array(selected)

def train_model(features, labels):
    """
    Train SVM classifier on selected features
    """

    max_seq_len = max([len(f) for f in features])

    # Pad Features to make features into a numpy array
    features_padded = pad_sequences(features, maxlen=max_seq_len, padding='post')
    features_array = np.array(features_padded)

     # Scale the padded numpy features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    svm = SVC()
    

    svm.fit(features_scaled, labels)
    return svm, scaler, max_seq_len

def evaluate_model(svm, scaler, max_seq_len, test_features, test_labels, test_file, steg_features, steg_files):
    """
    Evaluate model on test data 
    """

    # If Steg Features is empty return to menu and give warning.
    if not steg_features:
        print("Error: The Selected Folder Doesn't Contain WAV Files\n")
        menu()

    # Pad test features to convert into a numpy array
    test_padded = pad_sequences(test_features, maxlen=max_seq_len, padding='post')
    steg_padded = pad_sequences(steg_features, maxlen=max_seq_len, padding='post')

    # Scale padded test features to reduce effect of padding
    test_scaled = scaler.transform(test_padded)
    steg_scaled = scaler.transform(steg_padded)

    # Generate Prediction based on the test features.
    predictions = svm.predict(test_scaled)
    steg_predictions = svm.predict(steg_scaled)

    # Calculate Accuracy
    accuracy = np.mean(predictions == test_labels)

    # Remove Pre-existing test_output.csv
    if os.path.exists('test_output.csv'):
        os.remove('test_output.csv')

    # Remove Pre-existing steg_output.csv
    if os.path.exists('steg_output.csv'):
        os.remove('steg_output.csv')

    # Output predictions to test_output.csv
    with open('test_output.csv', 'w') as f:
        for i, prediction in enumerate(predictions):
            f.write(f"{test_file[i]}, {test_labels[i]}, {prediction}\n")

    # Check if steg_files is a directory or a file
    if os.path.isdir(steg_files):
        # If it's a directory, get all the .wav files in the directory
        steg_files_list = [os.path.join(steg_files, f) for f in os.listdir(steg_files) if f.endswith('.wav')]
    else:
        # If it's a file, put it in a list
        steg_files_list = [steg_files]

    # Output predictions to steg_output.csv
    steg_final = []
    with open('steg_output.csv', 'w') as f:
        for i, steg_prediction in enumerate(steg_predictions):
            steg_file = os.path.basename(steg_files_list[i])
            steg_string = f"{steg_file}, {steg_prediction}"
            steg_final.append(steg_string)
            f.write(f"{steg_string}\n")

    return accuracy, steg_final

def getStegFolderFilePath():
    """
    Takes a file/directory and makes sure there are WAV files
    """
    while True:
        path = input("Enter the path of the file or folder (or enter 'q' to quit): ")
        if path.lower() == 'q':
            path = "steganalysis_files"
            return path
        if os.path.exists(path):
            if os.path.isfile(path):
                # Check if the file is a .wav file
                if path.endswith('.wav'):
                    print(f"File {path} selected successfully.")
                    return path
                else:
                    print(f"Error: {path} is not a .wav file. Please select a .wav file.")
            elif os.path.isdir(path):
                print(f"Directory {path} selected successfully.")
                return path
        else:
            print(f"Error: {path} does not exist. Please try again.")
       

def option_four(training_files, training_labels, testing_files, testing_labels, steg_files): 
    """
    Menu option that initiates the testing and evluating of the model and chosen
    stegenalysis files
    """
    # Train the model
    features, labels = [], []
    df = pd.read_csv(training_labels, header=None)
    labels = np.array(df[1])
    for file in os.listdir(training_files):
        f = os.path.join(training_files, file)
        if os.path.isfile(f):
            samplerate, data = wavfile.read(f)
            data = data.astype(np.float16) / np.iinfo(data.dtype).max
            joint, cond, joint_diff, cond_diff = extract_features(data)
            selected = select_features((joint, cond, joint_diff, cond_diff))
            features.append(selected)
    svm, scaler, max_seq_len = train_model(features, labels)

    # Test the model
    test_features, test_labels, test_files = [], [], []
    df = pd.read_csv(testing_labels, header=None)
    test_labels = np.array(df[1])
    for test_file in os.listdir(testing_files):
        test_f = os.path.join(testing_files, test_file)
        if os.path.isfile(test_f):
            samplerate, data = wavfile.read(test_f)
            test_data = data.astype(np.float16) / np.iinfo(data.dtype).max
            joint, cond, joint_diff, cond_diff = extract_features(test_data)
            test_selected = select_features((joint, cond, joint_diff, cond_diff))
            test_features.append(test_selected)
            test_files.append(test_file)

    # Perform Steganalysis on Steg Files
    steg_features = []
    if os.path.isdir(steg_files):
        for steg_file in os.listdir(steg_files):
            steg_f = os.path.join(steg_files, steg_file)
            if os.path.isfile(steg_f) and steg_f.endswith('.wav'):
                samplerate, data = wavfile.read(steg_f)
                steg_data = data.astype(np.float16) / np.iinfo(data.dtype).max
                joint, cond, joint_diff, cond_diff = extract_features(steg_data)
                steg_selected = select_features((joint, cond, joint_diff, cond_diff))
                steg_features.append(steg_selected)
            else:
                print(f"Error: {steg_f} is not a WAV file.")
    else:
        if steg_files.endswith('.wav'):
            samplerate, data = wavfile.read(steg_files)
            steg_data = data.astype(np.float16) / np.iinfo(data.dtype).max
            joint, cond, joint_diff, cond_diff = extract_features(steg_data)
            steg_selected = select_features((joint, cond, joint_diff, cond_diff))
            steg_features.append(steg_selected)
    
    # Evaluate model accuracy and print it out.
    accuracy, steg_final = evaluate_model(svm, scaler, max_seq_len, test_features, test_labels, test_files, steg_features, steg_files)
    print(f"With {accuracy * 100}% confidence, here are the results!")
    for row in steg_final:
        filename, label = row.split(', ')
        if label == 0:
            print(f"{filename} has no hidden file.")
        else:
            print(f"{filename} has a hidden file.")

    # Return to the menu
    print()
    menu()

# Options for the menu.
def menu_options(choice):
    """
    Initiates menu options selected by user
    """

    # Lists files available for steganalysis 
    global training_files, training_labels, testing_files, testing_labels, lsbBitNum, steg_files
    if choice == "1":
        if os.path.exists(steg_files):
            if os.path.isfile(steg_files):
                print(f"File name: {os.path.basename(steg_files)}")
            elif os.path.isdir(steg_files):
                print(f"Contents of {os.path.basename(steg_files)}:")
                for item in os.listdir(steg_files):
                    print(f"- {item}")
        else:
            print(f"Error: {steg_files} does not exist.")
        menu()

    # Select LSB bit number
    elif choice == "2":
        while True:
            lsbBitNum = input("Enter a LSB bit value between 1 and 9 (9 selects ALL) (or enter 'q' to quit): ")
            if lsbBitNum.lower() == 'q':
                lsbBitNum = 0
                break
            elif lsbBitNum.isdigit() and 1 <= int(lsbBitNum) <= 9:
                break
            else:
                print(f"Error: {lsbBitNum} is not a valid number. Please try again.")
        if int(lsbBitNum) == 9:
            testing_files, testing_labels = "testing_data_folders/testing_data_all", "testing_labels_folder/TestingLabels_all.csv"
            training_files, training_labels = "training_data_folders/training_data_all", "training_labels_folder/TrainingLabels_all.csv"
        else:
            testing_files, testing_labels = f"testing_data_folders/testing_data_{lsbBitNum}lsb", f"testing_labels_folder/TestingLabels_{lsbBitNum}lsb.csv"
            training_files, training_labels = f"training_data_folders/training_data_{lsbBitNum}lsb", f"training_labels_folder/TrainingLabels_{lsbBitNum}lsb.csv"
        print(f"Selected Training Files: {training_files}")
        menu()

    # Select File to Analyze
    elif choice == "3":
        steg_files = getStegFolderFilePath()
        print()
        menu()

    # Train and Test Model
    elif choice == "4":
        if lsbBitNum == 0:
            print()
            print("Please enter the LSB value for training.")
            print()
            menu()
        option_four(training_files, training_labels, testing_files, testing_labels, steg_files)    

    # Exit Program
    elif choice == "5":
        sys.exit(0)

    # Invalid Option give warning then calls menu.
    else:
        print("Please only select an value between 1-5")
        print("Please try again\n")
        menu()

# Menu for the Program
def menu():
    """
    Lists menu options for the user
    """

    print("+-----------------------------------------------+")
    print("|************WAV File Steganalysis**************|")
    print("| 1. List Available Files for Steganalysis      |")
    print("| 2. Enter LSB Value for Training               |")
    print("| 3. Select File to Analyze                     |")
    print("| 4. Train and Test Model                       |")
    print("| 5. Exit Program                               |")
    print("+-----------------------------------------------+")

    choice = input("Please enter your choice: ")
    menu_options(choice)
    
# Main function calls menu
def main():

    menu()

main()