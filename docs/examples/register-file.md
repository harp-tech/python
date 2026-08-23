# Read a single register file

This example demonstrates how to load the binary data file of a **single** Harp register into a pandas DataFrame using `harp.data`. `parse_to_dataframe` decodes each frame against the register definition, so the result carries named columns and decoded enums.

!!! tip
    For a recorded session folder rather than one loose file, use [`open_dataset`](dataset.md), which resolves each register against the device schema so any of them can be read by class, by name, or by address.

<!--codeinclude-->
```python
[](./register_file.py)
```
<!--/codeinclude-->
