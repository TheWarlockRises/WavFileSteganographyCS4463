import os
import sys

folder_path = os.getcwd()
message_file = 'message.txt'
bits = sys.argv[1]

for file_name in os.listdir(folder_path):
    if file_name.endswith('.wav'):
        os.system(f'StegAudio32_v1.01.exe -hide -lsb -c {file_name} -m {message_file} -b {bits}')
