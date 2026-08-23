import numpy as np
from harp.data.synchronization import decode_clock_from_samples, decode_clock_from_transitions

# A non-Harp acquisition system recorded the downsampled Harp clock on one of its
# digital lines. `samples` is that line, sampled at the system's own rate.
samples = np.load("sync_line.npy")  # digital states; analog input needs a `threshold`
sample_rate = 30_000.0

# Decode it: one row per whole Harp second, keyed on the sample the packet was
# anchored on — the axis this system timestamps the rest of its data on too.
clock = decode_clock_from_samples(samples, sample_rate, baud_rate=1000.0)
print(clock.head())
#              Time
# Sample
# 37200   3806874.0
# 67203   3806875.0

# Anchors, so any of the system's timestamps — spikes, video frames, stimulus onsets —
# can be placed on the Harp axis. Interpolating between neighbouring anchors absorbs
# the drift between the two clocks.
spike_samples = np.load("spike_samples.npy")
harp_times = np.interp(spike_samples, clock.index, clock["Time"])

# Event-based systems report line transitions instead of a sampled waveform: a local
# time and the level the line took. Anchors then carry local seconds.
transitions = np.load("line_transitions.npy")
clock = decode_clock_from_transitions(transitions[:, 0], transitions[:, 1])
harp_times = np.interp(spike_samples / sample_rate, clock.index, clock["Time"])
