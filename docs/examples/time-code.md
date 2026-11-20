# Align to the Harp Time Code

Devices that are not part of the Harp bus run on their own clock, so their timestamps drift with respect to Harp time. The [Harp Time Code](https://harp-tech.org/protocol/TimeCode.html) exists so they can be aligned afterwards: a Harp device emits the current whole Harp second as a slow serial signal (typically 1 kbps) on a digital output, the non-Harp system records it on a spare digital or analog input, and `harp.data.timecode` decodes that recording back into Harp seconds.

Each packet starts exactly on the whole second it carries, so every decoded packet is an anchor: the sample (or local time) of its first falling edge, against the Harp second that begins there.

!!! note
    Every packet that frames cleanly is returned as decoded, including the rare one carrying a corrupt value, and a reset of the Harp clock mid-recording shows up as a jump in the anchors. The example below drops anchors that do not line up with the rest before fitting the offset and drift between the two clocks.

<!--codeinclude-->
```python
[](./time_code.py)
```
<!--/codeinclude-->
