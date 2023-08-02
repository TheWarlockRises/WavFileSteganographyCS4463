import numpy as np
import scipy.io
import os
from scipy.io import wavfile
from scipy.signal import savgol_filter
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt

def extract_features(audio):
    """
    Extract joint distribution and Markov features from 
    second order derivative of audio signal
    """
    
    # Take second order derivative
    derivative = savgol_filter(audio, window_length=5, polyorder=2, deriv=2)

    print(audio)
    print(derivative)
    
    # Rescale derivative values to range from -1 to 1
    derivative = 2 * (derivative - np.min(derivative)) / (np.max(derivative) - np.min(derivative)) - 1

    # Quantize derivative values to integers from -8 to 8
    derivative = np.round(derivative * 8).astype(int)

    # Build joint and conditional matrices
    joint_matrix = np.zeros((17,17))
    cond_matrix = np.zeros((17,17))
    N = len(derivative)
    for i in range(-8,9):
        for j in range(-8,9):
            joint_matrix[i+8,j+8] = np.sum((derivative[1:N-2] == i) & (derivative[2:N-1] == j)) / (N-3)
            cond_count = np.sum(derivative[2:-2] == i)
            if (cond_count > 0):
                cond_matrix[i+8,j+8] = np.sum((derivative[1:N-2] == i) & (derivative[2:N-1] == j)) / (N-2)
            else:
                cond_matrix[i+8,j+8] = 0          

    # print("Source min:", np.min(audio))
    # print("Source max:", np.max(audio))
    # print("Source unique values:", np.unique(audio))
    # print("Derivative min:", np.min(derivative))
    # print("Derivative max:", np.max(derivative))
    # print("Derivative unique values:", np.unique(derivative))
    #print(joint_matrix)
    #print(cond_matrix)
    
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
            if (mod_cond_count > 0):
                mod_cond[i+8,j+8] = np.sum((mod_deriv[2:-2] == i) & (mod_deriv[3:-1] == j)) / np.sum(mod_deriv[2:-2] == i)
            else:
                mod_cond[i+8,j+8] = 0
    
    #print(mod_joint)
    #print(mod_cond)

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
    threshold = np.percentile(joint, 90)
    
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

    print(np.array(selected))            
    return np.array(selected)

def menu():
    print("************WAV File Steganalysis**************")
    print()

    choice = input("""
                        1. List Available Files for Steganalysis
                        2. Analyze a File
                   
                        Please enter your choice: """)
    
    if choice == "1":
        for file in os.listdir():
            if file.endswith(".wav"):
                print(file)
    elif choice == "2":
        samplerate, data = wavfile.read('Test.wav')
        data = data.astype(np.float16) / np.iinfo(data.dtype).max
        joint, cond, joint_diff, cond_diff = extract_features(data)
        selected = select_features((joint, cond, joint_diff, cond_diff))
        print(selected)
    else:
        print("Please only select 1 or 2")
        print("Please try again")
        menu()

def main():
    menu()

main()