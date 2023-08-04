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

def evaluate_model(svm, scaler, max_seq_len, test_features, test_labels, test_file):
    """
    Evaluate model on test data 
    """

    # Pad test features to convert into a numpy array
    test_padded = pad_sequences(test_features, maxlen=max_seq_len, padding='post')

    # Scale padded test features to reduce effect of padding
    test_scaled = scaler.transform(test_padded)

    # Generate Prediction based on the test features.
    predictions = svm.predict(test_scaled)

    # Calculate Accuracy
    accuracy = np.mean(predictions == test_labels)

    # Remove Pre-existing test_output.csv
    if os.path.exists('test_output.csv'):
        os.remove('test_output.csv')

    # Output predictions to test_output.csv
    with open('test_output.csv', 'w') as f:
        for i, prediction in enumerate(predictions):
            f.write(f"{test_file[i]}, {prediction}\n")

    return accuracy

def option_two():
    # Train the model
    features, labels = [], []
    df = pd.read_csv('TrainingLabels.csv', header=None)
    labels = np.array(df[1])
    for file in os.listdir("training_data"):
        f = os.path.join("training_data", file)
        if os.path.isfile(f):
            samplerate, data = wavfile.read(f)
            data = data.astype(np.float16) / np.iinfo(data.dtype).max
            joint, cond, joint_diff, cond_diff = extract_features(data)
            selected = select_features((joint, cond, joint_diff, cond_diff))
            features.append(selected)
    svm, scaler, max_seq_len = train_model(features, labels)

    # Test the model
    test_features, test_labels, test_files = [], [], []
    df = pd.read_csv('TestingLabels.csv', header=None)
    test_labels = np.array(df[1])
    for test_file in os.listdir("testing_data"):
        test_f = os.path.join("testing_data", test_file)
        if os.path.isfile(test_f):
            samplerate, data = wavfile.read(test_f)
            test_data = data.astype(np.float16) / np.iinfo(data.dtype).max
            joint, cond, joint_diff, cond_diff = extract_features(test_data)
            test_selected = select_features((joint, cond, joint_diff, cond_diff))
            test_features.append(test_selected)
            test_files.append(test_file)
    
    # Evaluate model accuracy and print it out.
    accuracy = evaluate_model(svm, scaler, max_seq_len, test_features, test_labels, test_files)
    print("Accuracy: ", accuracy)

    # Return to the menu. Add spacing for aesthetics
    print()
    menu()

# Options for the menu.
def menu_options(choice):
    # Print all WAV file in the current directory maybe we can use this to select a directory for testing? Or we can just remove it.
    if choice == "1":
        for file in os.listdir():
            if file.endswith(".wav"):
                print(file)
        print()
        menu()    

    # Train and Test Model
    elif choice == "2":
        option_two()
    
    # Exit Program
    elif choice == "3":
        sys.exit(0)

    # Invalid Option give warning then calls menu.
    else:
        print("Please only select 1, 2, or 3")
        print("Please try again\n")
        menu()

# Menu for the Program
def menu():
    print("+-----------------------------------------------+")
    print("|************WAV File Steganalysis**************|")
    print("| 1. List Available Files for Steganalysis      |")
    print("| 2. Analyze Files                             |")
    print("| 3. Exit Program                               |")
    print("+-----------------------------------------------+")

    choice = input("Please enter your choice: ")
    menu_options(choice)
    
# Main function calls menu
def main():
    menu()

main()