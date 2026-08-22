# Subscribing to Events

This example demonstrates how to react to messages pushed by the device, e.g. unsolicited `Event` messages, without polling, using two subscription styles:

- `device.subscribe(register, handler)`, where the handler receives a `HarpMessage` typed by the payload of a single register.
- `device.subscribe_all(handler)`, a catch-all handler that receives the raw `HarpMessage` for every register.

Handlers run on a dedicated event thread, so they never block `read()` or `write()`. Both methods return a `Subscription`. Call `.unsubscribe()`, or use it as a context manager, to stop receiving events.

{% include-markdown "includes/serial-port.md" %}

<!--codeinclude-->
```python
[](./subscribing_to_events.py)
```
<!--/codeinclude-->
