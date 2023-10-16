from matplotlib.animation import FuncAnimation, FFMpegWriter
import sys  
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import scipy.signal
from scipy.io.wavfile import write
import multiprocess
from os import path
from pydub import AudioSegment 
from IPython import display 
import ffmpeg

# Functions

# Calculate FFT incrementally, based on setp size. 
# INDEX START, number, number, number..., INDEX END. Note: INDEX END - INDEX START = STEP_SIZE. 

def fft_calculator(signal, samplerate, index_start, index_end):
	signal = signal[int(index_start):int(index_end)]
	N = len(signal)
	frequency = np.linspace(0.0, samplerate/2, int(N/2))
	fft_data = rfft(signal)
	fft = 2/N * np.abs(fft_data[0:int (N/2)])
	return frequency, fft

# Defining the graph
def init():
	# F-Domain graph 
	axs[0].spines['right'].set_position('center')
	axs[0].spines['bottom'].set_position('zero')
	axs[0].spines['right'].set_color('none')
	axs[0].spines['top'].set_color('none')
	axs[0].xaxis.set_ticks_position('bottom')
	axs[0].yaxis.set_ticks_position('left')
	axs[0].set_xscale('log') 
	axs[0].set_ylim(0, 3500)
	# T-Domain graph 
	axs[1].spines['right'].set_position('center')
	axs[1].spines['bottom'].set_position('zero')
	axs[1].spines['right'].set_color('none')
	axs[1].spines['top'].set_color('none')
	axs[1].xaxis.set_ticks_position('bottom')
	axs[1].yaxis.set_ticks_position('left')
	axs[1].set_ylim(-6000, 6000)
	return ln,

# Updating index of samples
def animate(interval_change):
	axs[0].clear()
	axs[1].clear()
	init()
	index_start = interval_change
	index_end = index_start + step
	print(index_start, index_end, interval_change, step, len(signal[int(index_start):int(index_end)]))
	yf = fft_calculator(signal, samplerate, index_start, index_end)[1] # Second array gets you RANGE outputs in FFT function
	xf = fft_calculator(signal, samplerate, index_start, index_end)[0] # First array gets you DOMAIN outputs in FFT function
	axs[0].plot(xf, yf)
	axs[1].plot(signal[int(index_start):int(index_end)])
	return ln,

# NOTE: if signal is an mp3, convert to wav: 

# Signal input
Audio_File = "" # input file name here, as a string
Video_File = "" # Name of video, as a string 
Combined_File = "" # Name of video w/ audio, as as string 

# Access raw audio file
spf = wave.open(Audio_File, 'r')

# Extract raw audio from .wav file
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'int16')
samplerate = spf.getframerate()
print(samplerate)

# Filter the signal with a convolution:
windowSize = 100 
window = np.hanning(windowSize) 
window=window/window.sum()
confilt=np.convolve(window, signal, mode='valid')

# Time 
time = ((len(signal))/(2*samplerate))

# Frames per second
FPS = 24 

# Total number of frames 
Total_Frames = (FPS)*(time)

# Step size 
step = len(signal)/(Total_Frames)

# Print info for debugging purposes 
print(f"Step: {step}, Step (s): {1/FPS}, Total Length: {Total_Frames/FPS}")
print(len(signal), samplerate, time, step, FPS, Total_Frames) 

# Call graph 
fig, axs = plt.subplots(2)
ln, = axs[0].plot([],[])
axs[1].plot([],[])

# Animate graph in real time
anim = FuncAnimation(fig, animate, init_func=init, frames=np.arange(0, len(signal), step), 
	blit=True, interval=1000/FPS, repeat=False)

# Save video: 
writervideo=FFMpegWriter(fps=FPS)
anim.save(Video_File, writer=writervideo)

# Add audio to saved video 
input_video = ffmpeg.input(Video_File)
input_audio = ffmpeg.input(Audio_File)
ffmpeg.concat(input_video, input_audio, v=1, a=1).output(Combined_File).run(overwrite_output=True)
