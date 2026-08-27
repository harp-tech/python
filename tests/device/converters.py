from typing import Any

import numpy as np
from numpy.typing import NDArray

from harp.protocol import Converter


class DataConverter(Converter[np.int32]):
    """Maps two raw little-endian signed bytes to and from a numpy int32.

    Models interfaceType: int over a two-byte sub-region of the CustomMemberConverter payload.
    """

    init_kwarg_type = int

    def __init__(self) -> None:
        self._length = 2
        self.dtype = np.dtype((np.uint8, (self._length,)))

    def decode_scalar(self, view: np.generic) -> np.int32:
        return np.int32(int.from_bytes(bytes(np.asarray(view).tolist()), "little", signed=True))

    def decode_batch(self, view: NDArray[np.generic]) -> Any:
        return np.array(
            [
                int.from_bytes(bytes(np.asarray(r).tolist()), "little", signed=True)
                for r in np.atleast_2d(view)
            ],
            dtype=object,
        )

    def encode_into(self, view: NDArray[np.generic], value: np.int32) -> None:
        view[...] = np.frombuffer(
            int(value).to_bytes(self._length, "little", signed=True), dtype=np.uint8
        )
