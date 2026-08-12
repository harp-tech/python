import numpy as np
from harp.data.timecode import decode_timecode_from_samples, decode_timecode_from_transitions

# A non-Harp acquisition system recorded the Harp Time Code on one of its digital
# lines. `samples` is that line, sampled at the system's own rate.
samples = np.load("timecode_line.npy")  # digital states; analog input needs a `threshold`
sample_rate = 30_000.0

# Decode it: one row per whole Harp second, keyed on the sample where that second
# starts, the axis this system timestamps the rest of its data on too.
anchors = decode_timecode_from_samples(samples, sample_rate, baud_rate=1000.0)
print(anchors.head())
#              Time
# Sample
# 36000   3806874.0
# 66003   3806875.0

# Any of the system's timestamps (spikes, video frames, stimulus onsets) can now be
# placed on the Harp axis. Interpolating between neighbouring anchors absorbs the
# drift between the two clocks.
spike_samples = np.load("spike_samples.npy")
harp_times = np.interp(spike_samples, anchors.index, anchors["Time"])

# Event-based systems report line transitions instead of a sampled waveform: a local
# time and the level the line took. Anchors then carry local seconds.
transitions = np.load("line_transitions.npy")
anchors = decode_timecode_from_transitions(transitions[:, 0], transitions[:, 1])
harp_times = np.interp(spike_samples / sample_rate, anchors.index, anchors["Time"])
