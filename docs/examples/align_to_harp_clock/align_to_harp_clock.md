# Aligning Local Timestamps to the Harp Clock

Devices that are not part of the Harp bus run on their own clock, so their
timestamps drift with respect to Harp time. Some Harp clock emitters mirror the
[Synchronization Clock](https://harp-tech.org/protocol/SynchronizationClock.html)
on a digital output at a much lower baud rate — typically 1 kbps instead of
100 kbps — precisely so that such devices can record it on a spare digital (or
analog) input and be aligned afterwards.

This example decodes that recording back into Harp seconds, which local timestamps
can then be expressed against.

!!! note
    The decoded table is a set of anchors: local time → whole Harp second. How
    timestamps are placed between them is up to you — interpolating between
    neighbouring anchors absorbs the drift between the two clocks, whereas a global
    fit trades that away for noise rejection.

<!--codeinclude-->
```python
[](./align_to_harp_clock.py)
```
<!--/codeinclude-->
