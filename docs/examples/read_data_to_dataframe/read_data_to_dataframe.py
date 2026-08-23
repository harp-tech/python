from harp import data
from harp.device import core

# Parse the log file of a single register into a pandas DataFrame, one row per
# frame, one column per field. The register class tells `parse_to_dataframe` how
# to decode each frame, so the result carries named columns and decoded enums.
df = data.parse_to_dataframe(core.OperationControl, "OperationControl.bin")
print(df.head())

# When the frames are timestamped, which is the default, the Harp time becomes the
# DataFrame index, named "Time", holding float seconds from device start.
print(df.index.name, df.index[:3].to_list())

# `parse_to_dataframe` also accepts raw bytes or an open binary file object:
with open("OperationControl.bin", "rb") as f:
    df = data.parse_to_dataframe(core.OperationControl, f)

# To read registers from a recorded session folder, resolved against the device
# schema, use `harp.data.open_dataset`. See the "Reading a Whole Dataset Folder"
# example.
