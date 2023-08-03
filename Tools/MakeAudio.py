import math
import os
import random
import wave
import numpy as np

# Prompt the user to enter the number of files to generate
num_files = int(input('Enter the number of files to generate: '))

# Prompt the user to enter the sample rate range
sample_rate_range = input('Enter the sample rate range (e.g. 8000 48000): ')
sample_rate_range = tuple(map(int, sample_rate_range.split()))

# Prompt the user to enter the frequency range
frequency_range = input('Enter the frequency range (e.g. 20 20000): ')
frequency_range = tuple(map(int, frequency_range.split()))

# Set the range of possible durations
duration_range = (1, 10)

# Create the output directory if it doesn't exist
output_dir = 'Test_Audio'
os.makedirs(output_dir, exist_ok=True)

for i in range(num_files):
    # Generate a random sample rate, duration, and frequency
    sample_rate = random.randint(*sample_rate_range)
    duration = random.uniform(*duration_range)
    frequency = random.uniform(*frequency_range)

    # Calculate the number of samples
    num_samples = int(sample_rate * duration)

    # Generate the sine wave data
    data = np.sin(2 * np.pi * frequency * np.arange(num_samples) / sample_rate)

    # Convert the data to 16-bit integers
    data = np.int16(data * 32767)

    # Open a new WAV file for writing
    filename = f'Audio_{i}_16bit.wav'
    filepath = os.path.join(output_dir, filename)
    with wave.open(filepath, 'wb') as wav:
        # Set the parameters of the WAV file
        wav.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))

        # Write the audio data to the WAV file
        wav.writeframes(data.tobytes())

    print(f'Generated {filename} with sample rate {sample_rate}, duration {duration:.2f}s, and frequency {frequency:.2f}Hz')