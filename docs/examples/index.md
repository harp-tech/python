# Examples

This section contains examples for getting started with `harp`.

Working from a device schema:

- [Generating Registers from a Schema](./create_device_module/index.md) - compile a `device.yml` into a module of register classes at runtime with `create_device_module`.

Talking to a device:

- [Getting Device Info](./get_info/index.md) - connect to a Harp device and read its information.
- [Read and Write from Registers](./read_and_write_from_registers/index.md) - connect to a Harp device and read and write its registers.
- [Subscribing to Events](./subscribing_to_events/index.md) - react to messages pushed by the device without polling.

Reading recorded data:

- [Reading a Whole Dataset Folder](./read_dataset/index.md) - read registers from a recorded session folder into pandas DataFrames, decoded against the device schema.
- [Reading Data into a DataFrame](./read_data_to_dataframe/index.md) - decode the binary file of a single register into a pandas DataFrame.
