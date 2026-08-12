# Align to the Harp Time Code

Devices that are not part of the Harp bus run on their own clock, so their timestamps drift with respect to Harp time. The [Harp Time Code](https://harp-tech.org/protocol/TimeCode.html) exists so they can be aligned afterwards: a Harp device emits the current whole Harp second as a slow serial signal (typically 1 kbps) on a digital output, the non-Harp system records it on a spare digital or analog input, and `harp.data.timecode` decodes that recording back into Harp seconds.

Each packet starts exactly on the whole second it carries, so every decoded packet is an anchor: the sample (or local time) of its first falling edge, against the Harp second that begins there.

!!! note
    The decoded table is a set of anchors, local time → whole Harp second. How timestamps are placed between them is up to you: interpolating between neighbouring anchors absorbs the drift between the two clocks, whereas a global fit trades that away for noise rejection.

<!--codeinclude-->
```python
[](./time_code.py)
```
<!--/codeinclude-->
