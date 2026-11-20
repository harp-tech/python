import numpy as np
from harp.data.timecode import decode_timecode_from_samples, decode_timecode_from_transitions

# A non-Harp acquisition system recorded the Harp Time Code on one of its digital
# lines. `samples` is that line, sampled at the system's own rate.
samples = np.load("timecode_line.npy")  # digital or analog; the threshold is detected
sample_rate = 30_000.0

# Decode it: one row per whole Harp second, keyed on the sample where that second
# starts, the axis this system timestamps the rest of its data on too.
anchors = decode_timecode_from_samples(samples, sample_rate, baud_rate=1000.0)
print(anchors.head())
#              Time
# Sample
# 36000   3806874.0
# 66003   3806875.0

# Every cleanly framed packet is returned as decoded. Harp time minus local time stays
# nearly constant, so the rare corrupt value, or a reset of the Harp clock, stands out.
offset = anchors["Time"] - anchors.index / sample_rate
anchors = anchors[(offset - offset.median()).abs() < 1.0]

# A line through the anchors gives the offset and drift between the two clocks, and
# extends past the first and last anchors. Any of the system's timestamps (spikes,
# video frames, stimulus onsets) can then be placed on the Harp axis.
slope, intercept = np.polyfit(anchors.index, anchors["Time"], 1)
spike_samples = np.load("spike_samples.npy")
harp_times = intercept + slope * spike_samples

# Event-based systems report line transitions instead of a sampled waveform: a local
# time and the level the line took. Anchors then carry local seconds.
transitions = np.load("line_transitions.npy")
anchors = decode_timecode_from_transitions(transitions[:, 0], transitions[:, 1])
