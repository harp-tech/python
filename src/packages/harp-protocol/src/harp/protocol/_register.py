from abc import ABC, ABCMeta
from typing import Any, ClassVar, Generic, TypeVar, cast, final, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Sentinel

from ._builder import build_message_frame
from ._constants import (
    _DEFAULT_PORT,
    _HEADER_LEN,
    _MIN_FRAME_LEN,
    _TICK_PERIOD_S,
    _TIMESTAMP_FLAG,
    _TIMESTAMPED_PAYLOAD_OFFSET,
    _TS_MICROS_OFFSET,
)
from ._message import HarpMessage, HarpParseError
from ._message_type import MessageType, message_type_to_byte
from ._payload import (
    Batch,
    PayloadBase,
    PayloadFloat,
    PayloadFloatArray,
    PayloadS8,
    PayloadS8Array,
    PayloadS16,
    PayloadS16Array,
    PayloadS32,
    PayloadS32Array,
    PayloadS64,
    PayloadS64Array,
    PayloadU8,
    PayloadU8Array,
    PayloadU16,
    PayloadU16Array,
    PayloadU32,
    PayloadU32Array,
    PayloadU64,
    PayloadU64Array,
)
from ._payload_type import PayloadType, encode_payload_type

_MISSING = Sentinel("_MISSING")


def _encode_message_types(message_type: MessageType | ArrayLike, nrows: int) -> NDArray[np.uint8]:
    """Resolve a scalar/array ``message_type`` argument to N message-type bytes.

    A single :class:`MessageType` (error-bit aware) fills all frames; a scalar int
    is used verbatim; an array (e.g. the msgtype view from ``parse_bulk``, or a
    list of ``MessageType``/ints) becomes the per-frame bytes.
    """
    if isinstance(message_type, MessageType):
        return np.full(nrows, message_type_to_byte(message_type), dtype=np.uint8)
    values = np.asarray(message_type)
    if values.ndim == 0:
        return np.full(nrows, int(values.item()), dtype=np.uint8)
    return values.astype(np.uint8)


U = TypeVar("U")
_R = TypeVar("_R")
_AR = TypeVar("_AR", bound="RegisterBase[Any]")


class _LazyTimestamps:
    """Seconds + microseconds timestamp views, combined into float64 on first use.

    Combining the raw views costs an O(n) pass over every frame, two ``astype``
    casts plus a multiply-add, independent of the register payload, so eagerly
    computing it in ``parse_bulk`` taxes every call even when the caller never
    reads the timestamps. Deferring the combine until the array is actually
    accessed, and caching the result, avoids that cost in the common case where
    only the payload is needed.
    """

    __slots__ = ("_ts_s", "_ts_us", "_values")

    def __init__(self, ts_s: np.ndarray, ts_us: np.ndarray) -> None:
        self._ts_s = ts_s
        self._ts_us = ts_us
        self._values: np.ndarray | None = None

    def _resolve(self) -> np.ndarray:
        if self._values is None:
            out = np.multiply(self._ts_us, _TICK_PERIOD_S, dtype=np.float64)
            np.add(self._ts_s, out, out=out)
            self._values = out
        return self._values

    def __array__(self, dtype: "np.dtype | None" = None) -> np.ndarray:
        arr = self._resolve()
        return arr if dtype is None else arr.astype(dtype)

    def __len__(self) -> int:
        return len(self._ts_s)

    def __iter__(self):
        return iter(self._resolve())

    def __getitem__(self, item: Any) -> Any:
        return self._resolve()[item]

    def __repr__(self) -> str:
        return repr(self._resolve())


class _RegisterBaseMeta(ABCMeta):
    """The metaclass of every register, rendering its name and address."""

    def __repr__(cls) -> str:
        address = getattr(cls, "address", None)
        return super().__repr__() if address is None else f"<{cls.__name__} @{address}>"


def _require_no_address(cls: Any) -> None:
    address = getattr(cls, "address", None)
    if address is not None:
        raise TypeError(
            f"{cls.__name__} already declares address {address} and cannot be reassigned."
        )


class _ScalarRegisterMeta(_RegisterBaseMeta):
    """Calling a register base with an address creates a one-off subclass: ``RegisterU32(0x08)``."""

    def __call__(cls: "type[_R]", address: int) -> "type[_R]":
        _require_no_address(cls)
        return cast(
            "type[_R]",
            type(f"{cls.__name__}_{address:#04x}", (cls,), {"address": address}),
        )


class RegisterBase(ABC, Generic[U], metaclass=_RegisterBaseMeta):
    """Abstract base for all typed Harp registers.

    The generic parameter ``U`` is the static return type of :meth:`parse`, the
    user-facing value, *not* necessarily ``payload_class``, which is the wire
    encoding. The two coincide only for multi-member struct payloads:

    * scalar registers -> a numpy scalar, for example ``np.uint16``;
    * array registers  -> ``NDArray[...]`` of fixed length;
    * multi-member struct registers -> the payload class itself;
    * single-member registers that unwrap on parse -> the inner value type, for
      example ``RegisterBase[str]`` for DeviceName, ``RegisterBase[HarpVersion]``,
      or ``RegisterBase[ClockConfigurationFlags]`` for a whole-register
      ``BitMask`` or ``GroupMask``, even though each still has a ``payload_class``.

    Subclasses must define ``address``, ``payload_type``, and
    ``payload_class`` as ``ClassVar``s. ``length`` need not be declared: it is derived
    from ``payload_class``, and a subclass that does declare it keeps what it declared.
    """

    address: ClassVar[int]
    payload_type: ClassVar[PayloadType]
    payload_class: ClassVar[type[PayloadBase[Any]]]
    length: ClassVar[int | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Set ``length`` from ``payload_class``, so the two can never disagree."""
        super().__init_subclass__(**kwargs)
        has_payload_class = getattr(cls, "payload_class", None) is not None
        has_payload_type = getattr(cls, "payload_type", None) is not None
        if not (has_payload_class and has_payload_type):
            # We may want to remove this in the future (or guard against it).
            # It is here to allow the creation of intermediate abstract base
            # classes that don't declare a `payload_class` or `payload_type`.
            return
        cls.length = cls._derive_length()

    @classmethod
    def _derive_length(cls) -> int | None:
        """How many ``payload_type`` elements ``payload_class`` spans, ``None`` for one.

        The element width comes from ``payload_type`` rather than the payload's own
        ``_elem_dtype``, since that is what ``HarpMessage.decode`` measures the frame with
        and the two need not agree. A register declaring its own ``length`` has it checked
        against the payload but not otherwise trusted: the payload decides, and a
        sub-array payload counts as an array however few elements it spans, so arrayness
        comes from the layout rather than from anyone remembering to declare it.
        """
        payload_class: type[PayloadBase[Any]] = cls.payload_class
        payload_dtype: np.dtype = payload_class.payload_dtype
        itemsize: int = payload_dtype.itemsize
        element_size: int = np.dtype(cls.payload_type.value).itemsize
        count, remainder = divmod(itemsize, element_size)
        if remainder:
            raise TypeError(
                f"{cls.__name__}: payload {payload_class.__name__} spans {itemsize} bytes, "
                f"which is not a whole number of {cls.payload_type!r} elements "
                f"({element_size} byte(s))."
            )
        declared: int | None = cls.__dict__.get("length")
        if declared == 1:
            declared = None
        if declared is not None and declared != count:
            raise TypeError(
                f"{cls.__name__} declares length={declared!r} but payload "
                f"{payload_class.__name__} spans {count} {cls.payload_type!r} element(s)."
            )
        return count if count > 1 or payload_dtype.shape else None

    @classmethod
    def parse(cls, value: HarpMessage | bytes | bytearray | memoryview) -> U:
        """Parse a single message into the user-facing payload value.

        Struct payloads return a typed wrapper (descriptor access like
        ``payload.Channel0`` works). Anonymous payloads (scalar / array
        registers) return the raw numpy scalar or ndarray directly.
        """
        buf = value.payload_bytes if isinstance(value, HarpMessage) else value
        expected = cls.payload_class.payload_dtype.itemsize
        if len(buf) < expected:
            raise HarpParseError(
                f"{cls.__name__} reads {expected} payload bytes as {cls.payload_type!r} "
                f"but only {len(buf)} are available."
            )
        record = np.frombuffer(buf, dtype=cls.payload_class.payload_dtype, count=1)[0]
        return cast(U, cls.payload_class._unwrap(record))

    @classmethod
    def parse_bulk(
        cls,
        source: bytes | bytearray | memoryview,
        *,
        parse_timestamp: bool = True,
    ) -> "tuple[np.ndarray, _LazyTimestamps | None, np.ndarray | None, Batch[Any]]":
        """Parse a bulk buffer containing one or more frames of this register type. Returns (data, timestamps, msgtype_view, payload)."""
        # Returns (data, timestamps, msgtype_view, payload). ``data`` is
        payload_cls = cls.payload_class
        data = np.frombuffer(source, dtype=np.uint8)

        if len(data) == 0:
            # No frames, but still need to return a Batch with the right dtype.
            payload = payload_cls._from_array(np.empty(0, dtype=payload_cls.payload_dtype))
            return data, None, None, cast("Batch[Any]", payload)

        if len(data) < _MIN_FRAME_LEN:
            raise HarpParseError(
                f"{cls.__name__} reads frames of at least {_MIN_FRAME_LEN} bytes "
                f"but only {len(data)} are available."
            )

        stride = (
            int(data[1]) + 2
        )  # TODO this assumes all frames have the same length but we may want to revisit in the future.
        if len(data) < stride:
            raise HarpParseError(
                f"{cls.__name__} reads frames of {stride} bytes as declared by the first "
                f"length byte, but only {len(data)} are available."
            )
        nrows = len(data) // stride
        is_timestamped = bool(int(data[4]) & _TIMESTAMP_FLAG)
        payload_offset = _TIMESTAMPED_PAYLOAD_OFFSET if is_timestamped else _HEADER_LEN

        if is_timestamped and parse_timestamp:
            ts_s = np.ndarray(nrows, dtype="<u4", buffer=data, offset=_HEADER_LEN, strides=stride)
            ts_us = np.ndarray(
                nrows, dtype="<u2", buffer=data, offset=_TS_MICROS_OFFSET, strides=stride
            )
            timestamps = _LazyTimestamps(ts_s, ts_us)
        # TODO we may want to check if the timestamp is not present and users ask to be parsed. In that case we can either raise an error or return a nan-filled array
        else:
            timestamps = None

        msgtype_view = np.ndarray(nrows, dtype=np.uint8, buffer=data, offset=0, strides=stride)

        payload_arr = np.ndarray(
            nrows,
            dtype=payload_cls.payload_dtype,
            buffer=data,
            offset=payload_offset,
            strides=stride,
        )

        payload = payload_cls._from_array(payload_arr)
        return data, timestamps, msgtype_view, cast("Batch[Any]", payload)

    @classmethod
    def format_bulk(
        cls,
        values: PayloadBase | ArrayLike,
        *,
        timestamps: ArrayLike | None = None,
        message_type: MessageType | ArrayLike = MessageType.Event,
        port: int = _DEFAULT_PORT,
    ) -> NDArray[np.uint8]:
        """Build a flat buffer of N frames of this register type, the inverse of
        :meth:`parse_bulk`.

        ``values`` is a payload, either scalar or :class:`Batch`, or an ndarray of
        the ``payload_class.payload_dtype`` of the register. ``timestamps``, a
        length-N array of seconds, makes every frame timestamped. ``message_type``
        is one :class:`MessageType` for all frames, or a length-N array of
        message-type bytes or values, for example the ``msgtype`` view returned by
        ``parse_bulk``.
        """
        payload_cls = cls.payload_class
        record_dtype = payload_cls.payload_dtype
        itemsize = record_dtype.itemsize
        subdtype = record_dtype.subdtype
        if isinstance(values, PayloadBase):
            records = np.atleast_1d(np.asarray(values.payload_array))
        else:
            records = np.atleast_1d(np.asarray(values))
        if subdtype is not None and records.dtype != record_dtype:
            # An array register: convert against the element type and let the declared
            # shape decide the frame count, as `format` does for a single frame.
            element_dtype, shape = subdtype
            records = np.asarray(records, dtype=element_dtype)
            if records.shape[-len(shape) :] != shape:
                raise ValueError(
                    f"{cls.__name__}.format_bulk: values of shape {records.shape} do not end "
                    f"in the declared payload shape {shape}"
                )
            records = records.reshape(-1, *shape)
        elif (
            # Coerce the element type only for plain scalar payloads (e.g. an int
            # list for a scalar register). Struct records already carry the right
            # byte layout and must not be re-cast.
            records.dtype.names is None
            and records.dtype.subdtype is None
            and record_dtype.names is None
            and subdtype is None
            and records.dtype != record_dtype
        ):
            records = records.astype(record_dtype)
        nrows = len(records)
        flat = np.ascontiguousarray(records).tobytes()
        if len(flat) != nrows * itemsize:
            raise ValueError(
                f"{cls.__name__}.format_bulk: {len(flat)} payload bytes for {nrows} frames "
                f"is not a multiple of itemsize {itemsize}; check the values shape/dtype"
            )

        is_timestamped = timestamps is not None
        payload_offset = _TIMESTAMPED_PAYLOAD_OFFSET if is_timestamped else _HEADER_LEN
        stride = payload_offset + itemsize + 1  # trailing checksum byte

        buf = np.zeros((nrows, stride), dtype=np.uint8)
        buf[:, 0] = _encode_message_types(message_type, nrows)
        buf[:, 1] = stride - 2
        buf[:, 2] = cls.address
        buf[:, 3] = port
        buf[:, 4] = encode_payload_type(cls.payload_type, has_timestamp=is_timestamped)

        if is_timestamped:
            ts = np.atleast_1d(np.asarray(timestamps, dtype=np.float64))
            seconds = ts.astype(np.uint32)
            micros = np.round((ts - seconds.astype(np.float64)) / _TICK_PERIOD_S).astype(np.uint16)
            buf[:, _HEADER_LEN:_TS_MICROS_OFFSET] = np.frombuffer(
                seconds.astype("<u4").tobytes(), dtype=np.uint8
            ).reshape(nrows, 4)
            buf[:, _TS_MICROS_OFFSET:_TIMESTAMPED_PAYLOAD_OFFSET] = np.frombuffer(
                micros.astype("<u2").tobytes(), dtype=np.uint8
            ).reshape(nrows, 2)

        payload_bytes = np.frombuffer(flat, dtype=np.uint8).reshape(nrows, itemsize)
        buf[:, payload_offset : payload_offset + itemsize] = payload_bytes
        buf[:, -1] = buf[:, :-1].sum(axis=1, dtype=np.uint64).astype(np.uint8)
        return buf.reshape(-1)

    @overload
    @classmethod
    def format(
        cls,
        *,
        message_type: MessageType = MessageType.Read,
        timestamp: float | None = None,
        port: int = _DEFAULT_PORT,
    ) -> bytes: ...

    @overload
    @classmethod
    def format(
        cls,
        value: U,
        *,
        message_type: MessageType = MessageType.Write,
        timestamp: float | None = None,
        port: int = _DEFAULT_PORT,
    ) -> bytes: ...
    # We go with "U" for typing but it is worth noting that we accept "Any" below.
    # However for API ergonomics we want to keep the type hint for symmetry
    @final
    @classmethod
    def format(
        cls,
        value: Any = _MISSING,
        *,
        message_type: MessageType | None = None,
        timestamp: float | None = None,
        port: int = _DEFAULT_PORT,
    ) -> bytes:
        """Build a Harp frame for this register. No value gives a Read, a value gives a Write."""
        if value is _MISSING:
            mt = MessageType.Read if message_type is None else message_type
            return build_message_frame(
                mt, cls.address, cls.payload_type, port=port, timestamp=timestamp
            )
        else:
            mt = MessageType.Write if message_type is None else message_type
            if isinstance(value, PayloadBase):
                raw = value.payload_array.tobytes()
            elif isinstance(value, np.ndarray):
                raw = value.tobytes()
            else:
                # A bare high-level value (the symmetric counterpart of what
                # parse() returns): let the payload class encode it, so any
                # converter (e.g. a str via StringConverter) is applied.
                raw = cls.payload_class(value).payload_array.tobytes()
            return build_message_frame(
                mt, cls.address, cls.payload_type, raw, port=port, timestamp=timestamp
            )


class RegisterU8(RegisterBase[np.uint8], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a uint8 payload. ``parse()`` returns ``np.uint8``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = PayloadU8


class RegisterU16(RegisterBase[np.uint16], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a uint16 payload. ``parse()`` returns ``np.uint16``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U16
    payload_class = PayloadU16


class RegisterU32(RegisterBase[np.uint32], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a uint32 payload. ``parse()`` returns ``np.uint32``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U32
    payload_class = PayloadU32


class RegisterU64(RegisterBase[np.uint64], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a uint64 payload. ``parse()`` returns ``np.uint64``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U64
    payload_class = PayloadU64


class RegisterS8(RegisterBase[np.int8], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a int8 payload. ``parse()`` returns ``np.int8``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S8
    payload_class = PayloadS8


class RegisterS16(RegisterBase[np.int16], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a int16 payload. ``parse()`` returns ``np.int16``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S16
    payload_class = PayloadS16


class RegisterS32(RegisterBase[np.int32], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a int32 payload. ``parse()`` returns ``np.int32``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S32
    payload_class = PayloadS32


class RegisterS64(RegisterBase[np.int64], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a int64 payload. ``parse()`` returns ``np.int64``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S64
    payload_class = PayloadS64


class RegisterFloat(RegisterBase[np.float32], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a float32 payload. ``parse()`` returns ``np.float32``."""

    payload_type: ClassVar[PayloadType] = PayloadType.Float
    payload_class = PayloadFloat


class _ArrayRegisterMeta(_RegisterBaseMeta):
    """A base metaclass for array registers. Calling with address and length creates a concrete subclass: ``RegisterU16Array(0x28, length=3)``."""

    def __call__(cls: "type[_AR]", address: int, *, length: int) -> "type[_AR]":  # type: ignore[override, misc]
        _require_no_address(cls)
        base_payload = cls.payload_class  # type: ignore[attr-defined]
        # Anonymous payloads carry a plain (non-structured) dtype. The array
        # variant uses a sub-dtype (inner_dtype, (length,)) so a single buffer
        # element decodes directly to an ndarray of shape (length,).
        inner = base_payload.payload_dtype
        sub_dtype = np.dtype((inner, (length,)))
        concrete_payload = type(
            f"{base_payload.__name__}_{length}",
            (base_payload,),
            {"payload_dtype": sub_dtype},
        )
        return cast(
            "type[_AR]",
            type(
                f"{cls.__name__}_{address:#04x}",
                (cls,),
                {
                    "address": address,
                    "length": length,
                    "payload_class": concrete_payload,
                },
            ),
        )


class RegisterU8Array(RegisterBase[NDArray[np.uint8]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a uint8 array payload. It must be instantiated with a length: ``RegisterU8Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.uint8]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = PayloadU8Array


class RegisterU16Array(RegisterBase[NDArray[np.uint16]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a uint16 array payload. It must be instantiated with a length: ``RegisterU16Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.uint16]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U16
    payload_class = PayloadU16Array


class RegisterU32Array(RegisterBase[NDArray[np.uint32]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a uint32 array payload. It must be instantiated with a length: ``RegisterU32Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.uint32]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U32
    payload_class = PayloadU32Array


class RegisterU64Array(RegisterBase[NDArray[np.uint64]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a uint64 array payload. It must be instantiated with a length: ``RegisterU64Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.uint64]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U64
    payload_class = PayloadU64Array


class RegisterS8Array(RegisterBase[NDArray[np.int8]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a int8 array payload. It must be instantiated with a length: ``RegisterS8Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.int8]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S8
    payload_class = PayloadS8Array


class RegisterS16Array(RegisterBase[NDArray[np.int16]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a int16 array payload. It must be instantiated with a length: ``RegisterS16Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.int16]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S16
    payload_class = PayloadS16Array


class RegisterS32Array(RegisterBase[NDArray[np.int32]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a int32 array payload. It must be instantiated with a length: ``RegisterS32Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.int32]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S32
    payload_class = PayloadS32Array


class RegisterS64Array(RegisterBase[NDArray[np.int64]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a int64 array payload. It must be instantiated with a length: ``RegisterS64Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.int64]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S64
    payload_class = PayloadS64Array


class RegisterFloatArray(RegisterBase[NDArray[np.float32]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a float32 array payload. It must be instantiated with a length: ``RegisterFloatArray(0x28, length=3)``. ``parse()`` returns ``NDArray[np.float32]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.Float
    payload_class = PayloadFloatArray
