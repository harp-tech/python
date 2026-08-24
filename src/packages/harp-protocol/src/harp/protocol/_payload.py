import keyword
import enum
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    final,
    get_args,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self, Sentinel, dataclass_transform

from ._payload_converters import Converter as _Converter
from ._payload_converters import IdentityConverter as _IdentityConverter

NpStructT = TypeVar("NpStructT", bound=np.generic)
T = TypeVar("T")
E = TypeVar("E", bound=enum.IntEnum)
F = TypeVar("F", bound=enum.IntFlag)

_MISSING = Sentinel("_MISSING")
_DEFAULT_ELEMENT = np.dtype(np.uint8)


@dataclass(frozen=True, slots=True, eq=False)
class Column:
    """One column of a batched payload.

    ``data`` is a 1-D numpy array (one row per frame). When ``categories`` is
    not ``None`` the column is enum-backed: ``data`` holds integer category
    *codes* and ``categories`` the ordered labels, so a consumer can map codes
    to labels without copying.

    ``eq=False`` keeps identity comparison, since field-wise equality would hit
    the numpy ambiguous-truth-value error on the ``data`` array.

    ``name`` is ``None`` for an anonymous single value
    """

    name: str | None
    data: NDArray[Any]
    categories: Any | None = None


@dataclass(frozen=True)
class _FieldSlot:
    """One physical numpy field: its dtype and byte offset within the record."""

    dtype: np.dtype
    byte_offset: int


def _mask_trailing_zeros(mask: int) -> int:
    """Number of trailing zero bits in ``mask``, the right-shift that aligns a
    masked field to bit 0."""
    if mask == 0:
        return 0
    return (mask & -mask).bit_length() - 1


# ---------------------------------------------------------------------------
# Descriptors, scalar variants returning Python or 0-D types
# ---------------------------------------------------------------------------


class Field(Generic[T]):
    """Descriptor for a payload view decoded through a :class:`Converter`.

    Two modes, selected by ``mask``:

    * **Whole-element**, with ``mask=None`` as the default. The view reads
      ``converter.dtype.itemsize`` bytes starting at ``offset``, in base-element
      units as described in :class:`StructPayload`, and runs them through
      ``converter``. The converter owns its own ``dtype``, and so its own byte
      layout, and is independent of the base element type of the payload, so the
      same converter works under any register width.
    * **Masked sub-field**, with ``mask`` set. The raw value is extracted as
      ``(element & mask) >> shift`` from the *base element* of the payload at
      ``offset`` and then run through ``converter``, which dictates the output
      type. The right-shift is derived from the trailing-zero count of ``mask``.
      Several masked fields at the same offset share the element slot
      automatically, and may share it with a :class:`GroupMask` or
      :class:`BitMask` on the same word.

    ``offset`` defaults to ``0``. Omitting it suits a payload with a single
    member; when a payload has several distinct slots, each must declare an
    explicit ``offset=`` or the overlap check rejects the layout.
    """

    if TYPE_CHECKING:
        # Makes `field: T = Field(converter=...)` valid under @dataclass_transform without a
        # type-mismatch error. At runtime __new__ is not defined and a Field instance is
        # returned as normal.
        def __new__(  # type: ignore[misc]
            cls,
            converter: "_Converter[T]",
            *,
            mask: int | None = None,
            offset: int = 0,
            default: "T" = ...,
        ) -> "T": ...

    def __init__(
        self,
        converter: _Converter[T],
        *,
        mask: int | None = None,
        offset: int = 0,
        default: object = _MISSING,
    ) -> None:
        """Instantiates a new payload Field."""
        self._converter = converter
        self._mask = mask
        self._shift = _mask_trailing_zeros(mask) if mask is not None else 0
        self._offset = offset
        self._default = default
        # Numpy slot this field reads/writes; assigned in PayloadBase.__init_subclass__.
        self._slot: str = ""
        self._dtype: np.dtype = _DEFAULT_ELEMENT

    def _encode_value(self, value: Any) -> int:
        """Map a user value back to the integer to be masked + shifted into the slot
        (masked variant only)."""
        tmp = np.zeros((), dtype=self._converter.dtype)
        self._converter.encode_into(tmp, value)
        return int(tmp)

    def _bind_slot(self, slot: str, elem: np.dtype) -> None:
        """Bind this field to its numpy ``slot`` (called by ``_build_struct_dtype``)."""
        self._slot = slot
        if self._mask is not None:  # masked sub-field reads the base element
            self._dtype = elem

    @overload
    def __get__(self, obj: None, owner: object = None) -> "Field[T]": ...
    @overload
    def __get__(self, obj: "PayloadBase", owner: object = None) -> T: ...
    def __get__(self, obj: "PayloadBase | None", owner: object = None) -> Any:
        if obj is None:
            return self
        if self._mask is not None:
            raw = (obj._arr[self._slot] & self._mask) >> self._shift
            return self._converter.decode_scalar(self._converter.dtype.type(raw))
        return self._converter.decode_scalar(obj._arr[self._slot])  # pyright: ignore[reportArgumentType]

    def _to_batch(self) -> "_FieldBatch[T]":
        """Returns the metadata for the corresponding Batch type"""
        return _FieldBatch(
            converter=self._converter,
            mask=self._mask,
            slot=self._slot,
            dtype=self._dtype,
        )

    def _columns(
        self, arr: "NDArray[Any]", name: "str | None", *, decode_enums: bool, demux_bit_masks: bool
    ) -> "list[Column]":
        """Render this field as one or more batched :class:`Column`s (see ``to_columns``).

        ``name`` is the column name, or ``None`` for an anonymous root value (the
        consuming package decides the fallback label)."""
        if self._mask is not None:  # masked numeric sub-field
            raw = (arr[self._slot] & self._mask) >> self._shift
            return [Column(name, self._converter.decode_batch(raw.astype(self._converter.dtype)))]
        if not isinstance(self._converter, _IdentityConverter):  # whole-element, decoded
            return [Column(name, self._converter.decode_batch(arr[self._slot]))]
        sub = arr[self._slot]  # whole-element, raw passthrough
        if sub.ndim <= 1:
            return [Column(name, sub)]
        # sub-array -> one column per element; index is intrinsic identity, so a
        # nameless (root) array is positional, a named field is prefixed.
        width = int(np.prod(sub.shape[1:]))
        flat = sub.reshape(len(arr), width)
        label = (lambda i: str(i)) if name is None else (lambda i: f"{name}_{i}")
        return [Column(label(i), flat[:, i]) for i in range(flat.shape[1])]


def _build_enum_lookup(enum_cls: type[enum.IntEnum]) -> "tuple[list[str], np.ndarray]":
    """Category list + a code table mapping each raw enum value (``0..max member``) to its
    category index; a raw value with no member maps to -1."""
    members = list(enum_cls)
    categories = [m.name for m in members]
    max_val = max(int(m) for m in members)
    code_dtype = np.int8 if len(members) < 128 else np.int32
    code_lookup = np.full(max_val + 1, -1, dtype=code_dtype)
    for code, m in enumerate(members):
        code_lookup[int(m)] = code
    return categories, code_lookup


class GroupMask(Generic[E]):
    """Descriptor for a masked, shifted enum sub-field of a payload element.

    Syntactic sugar over a masked :class:`Field`: the raw value is extracted as
    ``(element & mask) >> shift`` and mapped strictly to an ``enum.IntEnum`` member,
    and an unknown code raises. ``enum=`` is required. For masked *numeric* fields use
    ``Field(converter=..., mask=...)`` instead.

    The right-shift is always derived from the trailing-zero count of ``mask``, so the
    field aligns to bit 0, and ``offset`` defaults to ``0``. The element width and
    storage slot are derived from the base element type of the payload, so several
    masked fields at the same offset share storage automatically.
    """

    if TYPE_CHECKING:
        # enum variant -> the field type is the enum
        def __new__(  # type: ignore[misc]  # noqa: E704
            cls, *, mask: int, enum: "type[E]", offset: int = 0, default: "E" = ...
        ) -> "E": ...

    def __init__(
        self,
        *,
        mask: int,
        enum: type[E],
        offset: int = 0,
        default: object = _MISSING,
    ) -> None:
        """Instantiates a GroupMask field for the payload"""
        if enum is None:
            raise TypeError(
                "GroupMask requires 'enum'; use Field(converter=..., mask=...) for "
                "masked numeric sub-fields"
            )
        self._mask = mask
        self._shift = _mask_trailing_zeros(mask)
        self._enum = enum
        self._offset = offset
        self._default = default
        # Numpy slot assigned in PayloadBase.__init_subclass__.
        self._slot: str = ""
        self._dtype: np.dtype = _DEFAULT_ELEMENT
        self._categories, self._code_lookup = _build_enum_lookup(enum)
        self._lookup_safe = (mask >> self._shift) < len(self._code_lookup)

    def _decode_raw(self, raw: Any) -> Any:
        """Map an extracted masked and shifted integer to its enum member, preserving an
        undefined code as its raw int. Decoding is permissive, like the unchecked enum
        cast in C#."""
        value = int(raw)
        try:
            return self._enum(value)
        except ValueError:
            return value

    def _encode_value(self, value: Any) -> int:
        """Map a user value back to the integer to be masked + shifted into the slot."""
        return int(value)

    def _bind_slot(self, slot: str, elem: np.dtype) -> None:
        """Bind this group mask to its shared numpy ``slot``."""
        self._slot = slot
        self._dtype = elem

    @overload
    def __get__(self, obj: None, owner: object = None) -> "GroupMask[E]": ...
    @overload
    def __get__(self, obj: "PayloadBase", owner: object = None) -> E: ...
    def __get__(self, obj: "PayloadBase | None", owner: object = None) -> Any:
        if obj is None:
            return self
        raw = (obj._arr[self._slot] & self._mask) >> self._shift
        return self._decode_raw(raw)

    def _to_batch(self) -> "_GroupMaskBatch[E]":
        """Returns the metadata for the corresponding batch type"""
        return _GroupMaskBatch(
            self._mask,
            self._enum,
            slot=self._slot,
            dtype=self._dtype,
        )

    def _columns(
        self, arr: "NDArray[Any]", name: "str | None", *, decode_enums: bool, demux_bit_masks: bool
    ) -> "list[Column]":
        """One enum column: category codes and labels under ``decode_enums``, or raw codes."""
        raw = (arr[self._slot] & self._mask) >> self._shift
        if not decode_enums:
            return [Column(name, raw)]
        lookup = self._code_lookup
        # ``_lookup_safe`` means the raw range of the field fits the table, so the bounds
        # guard is skipped. An in-range gap still maps to -1, so the undefined branch
        # below runs regardless.
        if self._lookup_safe:
            codes = lookup[raw]
        else:
            codes = np.where(raw < len(lookup), lookup.take(raw, mode="clip"), -1)
        undefined = codes < 0
        if not undefined.any():
            return [Column(name, codes, self._categories)]
        # An undefined code, either an in-range gap or a value past the range of the
        # enum, is kept as its raw integer and becomes an extra category, matching the
        # scalar decode and the unchecked cast in C#.
        codes = codes.astype(np.intp)
        extras = np.unique(raw[undefined])
        codes[undefined] = len(self._categories) + np.searchsorted(extras, raw[undefined])
        return [Column(name, codes, list(self._categories) + extras.tolist())]


class BitMask(Generic[F]):
    """Descriptor for a masked ``enum.IntFlag`` view of a payload element.

    The flag counterpart of :class:`GroupMask`: the raw value is extracted as
    ``element & mask`` and mapped to an ``enum.IntFlag`` member. Decoding is
    *permissive*, so combined flag values such as ``A | B`` are valid, matching the
    unchecked cast of the C# generator. ``enum=`` is required and must be an
    ``IntFlag`` subclass.

    Unlike :class:`GroupMask` there is **no shift**: ``IntFlag`` member values are
    absolute bit positions, so the flags are read and written in place. ``mask``
    defaults to the full base element, the common whole-register bitMask case, and
    may be narrowed to embed a flag set inside a wider element. The element width
    and storage slot are derived from the base element type of the payload, so
    several masked fields at the same offset share storage automatically.
    """

    if TYPE_CHECKING:
        # flag variant -> the field type is the IntFlag
        def __new__(  # type: ignore[misc]  # noqa: E704
            cls, *, enum: "type[F]", mask: int | None = None, offset: int = 0, default: "F" = ...
        ) -> "F": ...

    def __init__(
        self,
        *,
        enum: type[F],
        mask: int | None = None,
        offset: int = 0,
        default: object = _MISSING,
    ) -> None:
        """Instantiates a BitMask field for the payload"""
        if enum is None:
            raise TypeError("BitMask requires 'enum' (an enum.IntFlag subclass)")
        # mask=None is resolved to the full base element in _build_struct_dtype.
        self._mask = mask
        self._shift = 0  # IntFlag members are absolute bit positions; never shifted
        self._enum = enum
        self._offset = offset
        self._default = default
        # Numpy slot assigned in PayloadBase.__init_subclass__.
        self._slot: str = ""
        self._dtype: np.dtype = _DEFAULT_ELEMENT

    def _decode_raw(self, raw: Any) -> Any:
        """Map an extracted (masked) integer to its IntFlag value (permissive)."""
        return self._enum(int(raw))

    def _encode_value(self, value: Any) -> int:
        """Map a user value back to the integer to be masked into the slot."""
        return int(value)

    def _bind_slot(self, slot: str, elem: np.dtype) -> None:
        """Bind this flag mask to its shared numpy ``slot``; default the mask to the full element."""
        self._slot = slot
        self._dtype = elem
        if self._mask is None:
            self._mask = (1 << (elem.itemsize * 8)) - 1

    @overload
    def __get__(self, obj: None, owner: object = None) -> "BitMask[F]": ...
    @overload
    def __get__(self, obj: "PayloadBase", owner: object = None) -> F: ...
    def __get__(self, obj: "PayloadBase | None", owner: object = None) -> Any:
        if obj is None:
            return self
        raw = obj._arr[self._slot] & self._mask
        return self._decode_raw(raw)

    def _to_batch(self) -> "_BitMaskBatch[F]":
        """Returns the metadata for the corresponding batch type"""
        assert self._mask is not None  # _bind_slot ensures every masked field has a mask
        return _BitMaskBatch(
            self._mask,
            self._enum,
            slot=self._slot,
            dtype=self._dtype,
        )

    def _columns(
        self, arr: "NDArray[Any]", name: "str | None", *, decode_enums: bool, demux_bit_masks: bool
    ) -> "list[Column]":
        """A single raw-integer column, or (``demux_bit_masks``) one bool column per flag member."""
        assert self._mask is not None  # _bind_slot ensures every masked field has a mask
        raw = arr[self._slot] & self._mask
        if not demux_bit_masks:
            return [Column(name, raw)]
        # One boolean column per flag member that fits the field.
        return [
            Column(member.name, (raw & int(member)) != 0)
            for member in self._enum
            if not (int(member) & ~self._mask)  # skip bits that can't fit the field
        ]


# ---------------------------------------------------------------------------
# Descriptors, batch variants returning ndarray views
# These are mostly used for batch operations like `to_dataframe`
# ---------------------------------------------------------------------------


class _FieldBatch(Generic[T]):
    """Same as _Field but returns an NDArray view for batch payloads rather than a scalar value."""

    def __init__(
        self,
        *,
        converter: _Converter[T],
        mask: int | None = None,
        slot: str = "",
        dtype: "np.dtype | str | type" = np.uint8,
    ) -> None:
        self._converter = converter
        self._mask = mask
        self._shift = _mask_trailing_zeros(mask) if mask is not None else 0
        self._slot = slot
        self._dtype = np.dtype(dtype)

    @overload
    def __get__(self, obj: None, owner: object = None) -> "_FieldBatch[T]": ...
    @overload
    def __get__(self, obj: "PayloadBase", owner: object = None) -> "NDArray[Any]": ...
    def __get__(self, obj: "PayloadBase | None", owner: object = None) -> Any:
        if obj is None:
            return self
        if self._mask is not None:
            raw = (obj._arr[self._slot] & self._mask) >> self._shift
            return self._converter.decode_batch(raw.astype(self._converter.dtype))
        return self._converter.decode_batch(obj._arr[self._slot])


class _BitMaskBatch(Generic[F]):
    """Same as BitMask but returns an NDArray view for batch payloads rather than a scalar value."""

    def __init__(
        self,
        mask: int,
        enum: type[F],
        *,
        slot: str = "",
        dtype: "np.dtype | str | type" = np.uint8,
    ) -> None:
        self._mask = mask
        self._shift = 0
        self._enum = enum
        self._slot = slot
        self._dtype = np.dtype(dtype)

    @overload
    def __get__(self, obj: None, owner: object = None) -> "_BitMaskBatch[F]": ...
    @overload
    def __get__(self, obj: "PayloadBase", owner: object = None) -> "NDArray[Any]": ...
    def __get__(self, obj: "PayloadBase | None", owner: object = None) -> Any:
        if obj is None:
            return self
        return obj._arr[self._slot] & self._mask


class _GroupMaskBatch(Generic[E]):
    """Same as GroupMask but returns an NDArray view for batch payloads rather than a scalar value."""

    def __init__(
        self,
        mask: int,
        enum: type[E],
        *,
        slot: str = "",
        dtype: "np.dtype | str | type" = np.uint8,
    ) -> None:
        self._mask = mask
        self._shift = _mask_trailing_zeros(mask)
        self._enum = enum
        self._slot = slot
        self._dtype = np.dtype(dtype)
        self._categories, self._code_lookup = _build_enum_lookup(enum)

    @overload
    def __get__(self, obj: None, owner: object = None) -> "_GroupMaskBatch[E]": ...
    @overload
    def __get__(self, obj: "PayloadBase", owner: object = None) -> "NDArray[Any]": ...
    def __get__(self, obj: "PayloadBase | None", owner: object = None) -> Any:
        if obj is None:
            return self
        # Enum batches return raw integer codes (see to_columns for label decoding).
        return (obj._arr[self._slot] & self._mask) >> self._shift


_PT = TypeVar("_PT", bound="PayloadBase[Any]", covariant=True)
_MISSING_INIT = Sentinel("_MISSING_INIT")


class Batch(Protocol[_PT]):
    """Alias type for batched payloads so we can have nice type-hinting
    for batch operations like `read_frames` and `to_dataframe`.

    Statically, ``Batch[P]`` is a distinct type from ``P`` so the type
    checker knows ``read_frames`` returns an ndarray-shaped view rather
    than a single record. At runtime, the value is the auto-derived
    ``P._PayloadBatchType`` sibling whose descriptors return ``NDArray`` views.

    Per-field dtype precision is intentionally dropped, with every declared
    field reporting ``NDArray[Any]``, to keep ``RegisterBase[P]``
    parameterized by a single TypeVar.
    """

    payload_array: "NDArray[Any]"
    value: "NDArray[Any]"

    def __len__(self) -> int: ...  # type: ignore[empty-body]

    def payload_as_columns(  # type: ignore[empty-body]
        self, *, decode_enums: bool = True, demux_bit_masks: bool = False
    ) -> "list[Column]": ...

    def __getattr__(self, name: str) -> "NDArray[Any]": ...  # type: ignore[empty-body]


# Descriptor type tuples used for isinstance checks over payload declarations.
_SCALAR_DECLARATION_TYPES = (Field, GroupMask, BitMask)
_BATCH_DECLARATION_TYPES = (_FieldBatch, _GroupMaskBatch, _BitMaskBatch)
_DECLARATION_TYPES = _SCALAR_DECLARATION_TYPES + _BATCH_DECLARATION_TYPES


_RESERVED_FIELD_PREFIXES = ("_", "payload_")
"""Every member the payload classes own carries one of these prefixes, so a field name
is barred from them rather than from a list of the members themselves. Dunders are
exempt because ``__value__`` is how a single-slot payload declares its root field."""


def _reserved_field_reason(name: str) -> "str | None":
    """Returns why ``name`` cannot be a payload field, or ``None`` when it can."""
    if name.startswith("__") and name.endswith("__"):
        return None
    if not name.isidentifier():
        return "is not a valid Python identifier"
    if keyword.iskeyword(name):
        return "is a Python keyword"
    for prefix in _RESERVED_FIELD_PREFIXES:
        if name.startswith(prefix):
            return f"starts with {prefix!r}, which is reserved for payload members"
    return None


def _batch_init_disabled(self: "PayloadBase", *args: object, **kwargs: object) -> None:
    raise TypeError(
        f"{type(self).__name__} is a Batch payload; construct it via "
        f"from_array()/from_buffer() (or use its scalar twin "
        f"{type(self)._scalar_cls.__name__!s})."
    )


def _resolve_element_dtype(cls: type) -> np.dtype:
    """Resolve the base element dtype from the ``StructPayload[...]`` type arg.

    Only used for offset-to-byte arithmetic and the masked-read integer width.
    Defaults to uint8 (byte) when the payload is not parameterized, which makes
    byte-offset layouts of heterogeneous consecutive fields work out of the box.
    """
    for base in getattr(cls, "__orig_bases__", ()):
        for arg in get_args(base):
            if isinstance(arg, TypeVar):
                continue
            try:
                return np.dtype(arg)
            except TypeError:
                continue
    return _DEFAULT_ELEMENT


def _validate_mask_fits(cls: type, name: str, mask: int, elem: np.dtype) -> None:
    if mask < 0 or mask >= (1 << (elem.itemsize * 8)):
        raise TypeError(
            f"{cls.__name__}: mask {mask:#x} on {name!r} does not fit the base "
            f"element {elem} ({elem.itemsize} byte(s))"
        )


def _validate_no_overlap(cls: type, slots: "dict[str, _FieldSlot]", itemsize: int) -> None:
    """Validate that declared fields do not overlap and fit within the payload itemsize. Overlap is only
    allowed for masked fields sharing the same slot."""
    spans = sorted(
        (slot.byte_offset, slot.byte_offset + slot.dtype.itemsize, name)
        for name, slot in slots.items()
    )
    for start, end, name in spans:
        if end > itemsize:
            raise TypeError(
                f"{cls.__name__}: field {name!r} ends at byte {end}, beyond itemsize "
                f"{itemsize}; declare an explicit length="
            )
    for (prev_start, prev_end, prev_name), (start, end, name) in zip(spans, spans[1:]):
        if start < prev_end:
            raise TypeError(
                f"{cls.__name__}: fields {prev_name!r} and {name!r} overlap "
                f"(bytes [{prev_start},{prev_end}) vs [{start},{end})); masked sub-fields "
                f"of the same element must share an offset"
            )


def _build_struct_dtype(
    cls: "type[PayloadBase]",
    declarations: "list[tuple[str, Field | GroupMask | BitMask]]",
    length: int | None,
) -> np.dtype:
    """Build the numpy structured dtype from field declarations.

    Each descriptor binds itself to its numpy slot via ``_bind_slot``. This
    function resolves only cross-field layout: which masked fields share a slot,
    plus offsets, overlap, and itemsize."""
    elem = cls._elem_dtype
    elem_size = elem.itemsize
    slots: dict[str, _FieldSlot] = {}
    mask_slot_by_byte_offset: dict[int, str] = {}

    for attr_name, val in declarations:
        byte_offset = val._offset * elem_size
        # A plain Field with no mask= is the only whole-element view. Everything else,
        # a GroupMask, a BitMask, or a Field with mask=, is a masked sub-field. The
        # isinstance form lets the type checker narrow `val` to access ``_converter``.
        if isinstance(val, Field) and val._mask is None:  # whole-element Field, own slot
            if attr_name in slots:
                raise TypeError(f"{cls.__name__}: duplicate field name {attr_name!r}")
            val._bind_slot(attr_name, elem)
            slots[attr_name] = _FieldSlot(val._converter.dtype, byte_offset)
        else:  # masked sub-field, shares the base-element slot at its offset
            owner = mask_slot_by_byte_offset.setdefault(byte_offset, attr_name)
            val._bind_slot(owner, elem)
            assert val._mask is not None  # _bind_slot ensures every masked field has a mask
            _validate_mask_fits(cls, attr_name, val._mask, elem)
            if owner == attr_name:
                slots[attr_name] = _FieldSlot(elem, byte_offset)

    if length is not None:
        itemsize = length * elem_size
    else:
        itemsize = max(slot.byte_offset + slot.dtype.itemsize for slot in slots.values())
    _validate_no_overlap(cls, slots, itemsize)
    return np.dtype(
        {
            "names": list(slots),
            "formats": [slot.dtype for slot in slots.values()],
            "offsets": [slot.byte_offset for slot in slots.values()],
            "itemsize": itemsize,
        }
    )


class PayloadBase(Generic[NpStructT]):
    """Base class for typed Harp register payloads.

    A subclass declares fields via the scalar descriptors above.
    ``__init_subclass__`` auto-derives a ``Batch`` sibling subclass with the
    same dtype but each descriptor swapped to a Batch variant returning an
    ``NDArray`` view. ``from_array`` routes by ``ndim`` so callers never need
    to mention the Batch class explicitly: 0-D records stay scalar, 1-D
    buffers become Batch.
    """

    # Structured numpy dtype describing the memory layout of a single payload record.
    payload_dtype: ClassVar[np.dtype]
    # Field names shown in __repr__ and used as the column order.
    _repr_fields: ClassVar[tuple[str, ...]]
    # The scalar twin of this class (identity for scalar classes, points to scalar from Batch).
    _scalar_cls: ClassVar["type[PayloadBase]"]
    # The batch twin of this class (identity until the Batch sibling is generated).
    _batch_cls: ClassVar["type[PayloadBase]"]
    # Cached map of attribute name to default value for fields that declare one.
    _defaults: ClassVar[dict[str, Any]]
    # Auto-generated sibling class whose descriptors return NDArray views instead of scalars.
    _PayloadBatchType: ClassVar["type[PayloadBase]"]
    # Base element dtype (from the ``StructPayload[...]`` type arg); governs offset
    # arithmetic and the integer width used for masked reads. Defaults to uint8.
    _elem_dtype: ClassVar[np.dtype] = _DEFAULT_ELEMENT
    # The ``__value__`` field of an AnonymousPayload root, else None: ``parse`` unwraps to it.
    _single_member: ClassVar[str | None] = None
    # The underlying numpy array holding one (0-D) or many (1-D) payload records.
    _arr: NDArray[NpStructT]

    def __init__(self, *args: object, **kwargs: object) -> None:
        cls = type(self)
        names = self.payload_dtype.names
        if names is None:
            raise TypeError(f"{type(self).__name__}.dtype has no named fields")

        # TODO this is just to allow; PayloadU16(foo) syntax. Not sure if it is worth it?
        if args and kwargs:
            raise TypeError(
                f"{cls.__name__}() does not accept positional and keyword args together"
            )
        if args:
            if len(args) != 1 or len(names) != 1:
                raise TypeError(f"{cls.__name__}() takes exactly one positional argument")
            kwargs = {names[0]: args[0]}

        defaults = cls._defaults
        if defaults:
            merged = {k: v for k, v in defaults.items() if k not in kwargs}
            if merged:
                merged.update(kwargs)
                kwargs = merged

        arr = np.zeros((), dtype=self.payload_dtype)

        # Route each kwarg by its descriptor kind, not by whether its name happens
        # to match a numpy slot, since masked descriptors may share a slot whose
        # name collides with the attribute name of the first masked field.
        for attr_name, value in kwargs.items():
            desc = cls._mro_descriptor(attr_name)
            if isinstance(desc, Field) and desc._mask is None:  # whole-element Field
                desc._converter.encode_into(arr[desc._slot], value)
            elif isinstance(desc, (GroupMask, BitMask, Field)):
                # masked sub-field -> encode, shift, merge into the shared slot
                mask = desc._mask
                assert mask is not None  # invariant for masked descriptors
                mask_in_dtype = np.array(mask, dtype=desc._dtype)
                int_val = desc._encode_value(value)
                shifted = np.array((int_val << desc._shift) & mask, dtype=desc._dtype)
                arr[desc._slot] = (arr[desc._slot] & ~mask_in_dtype) | shifted
            elif attr_name in names:
                arr[attr_name] = value  # raw slot with no descriptor
            else:
                raise TypeError(f"{cls.__name__}() got unexpected kwarg: {attr_name!r}")

        self._arr = arr

    @classmethod
    def _mro_descriptor(cls, name: str) -> "Field[Any] | GroupMask[Any] | BitMask[Any] | None":
        for klass in cls.__mro__:
            if name in klass.__dict__:
                return klass.__dict__[name]
        return None

    @classmethod
    def _collect_defaults(cls) -> "dict[str, Any]":
        """Collect all members with defined default values"""
        out: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            for attr, val in klass.__dict__.items():
                if isinstance(val, (Field, GroupMask, BitMask)) and val._default is not _MISSING:
                    out[attr] = val._default
        return out

    @classmethod
    def _collect_repr_fields(cls) -> "tuple[str, ...]":
        """Declared attribute names in MRO + definition order.

        Covers plain ``Field``s and masked sub-fields alike (a single dtype slot
        may back several bitfields, so this enumerates declarations rather than
        ``dtype.names``).
        """
        names: list[str] = []
        for klass in reversed(cls.__mro__):
            for attr, val in klass.__dict__.items():
                if isinstance(val, _SCALAR_DECLARATION_TYPES) and attr not in names:
                    names.append(attr)
        return tuple(names)

    def __init_subclass__(
        cls,
        *,
        _batch_of: "type[PayloadBase] | None" = None,
        length: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)

        if _batch_of is not None:
            # Auto-generated Batch sibling: borrow dtype/_repr_fields from its
            # scalar twin and wire the pointers between scalar and batch.
            cls.payload_dtype = _batch_of.payload_dtype
            cls._repr_fields = _batch_of._repr_fields
            cls._elem_dtype = _batch_of._elem_dtype
            cls._single_member = _batch_of._single_member
            cls._scalar_cls = _batch_of
            cls._batch_cls = cls
            _batch_of._batch_cls = cls
            return

        cls._elem_dtype = _resolve_element_dtype(cls)
        cls._single_member = None

        for name, val in cls.__dict__.items():
            if isinstance(val, _DECLARATION_TYPES):
                reason = _reserved_field_reason(name)
                if reason is not None:
                    raise TypeError(f"{cls.__name__}: field name {name!r} {reason}")

        own_declarations = [
            (name, val)
            for name, val in cls.__dict__.items()
            if isinstance(val, _SCALAR_DECLARATION_TYPES)
        ]

        if own_declarations:
            cls.payload_dtype = _build_struct_dtype(cls, own_declarations, length)
            # Only an AnonymousPayload root (its lone __value__ field) unwraps on
            # parse; a StructPayload always returns the wrapper, never auto-unwraps.
            if getattr(cls, "_root", False):
                cls._single_member = own_declarations[0][0]

        if "_repr_fields" not in cls.__dict__:
            cls._repr_fields = cls._collect_repr_fields()

        cls._scalar_cls = cls
        cls._batch_cls = cls  # rebound below once Batch is generated

        if hasattr(cls, "payload_dtype"):
            batch_attrs: dict[str, Any] = {"__init__": _batch_init_disabled}
            for name, val in cls.__dict__.items():
                if isinstance(val, _SCALAR_DECLARATION_TYPES):
                    batch_attrs[name] = val._to_batch()
            cls._PayloadBatchType = type(
                f"{cls.__name__}Batch",
                (cls,),
                batch_attrs,
                _batch_of=cls,
            )

        cls._defaults = cls._collect_defaults()

    @classmethod
    def _from_array(cls, arr: "np.ndarray") -> Self:
        target = cls._scalar_cls if arr.ndim == 0 else cls._batch_cls
        obj = target.__new__(target)
        obj._arr = arr
        return obj  # type: ignore[return-value]

    @classmethod
    def payload_from_buffer(cls, buf: bytes | bytearray | memoryview) -> Self:
        arr = np.frombuffer(buf, dtype=cls.payload_dtype)
        return cls._from_array(arr[0] if len(arr) == 1 else arr)

    @property
    def payload_array(self) -> NDArray[NpStructT]:
        return self._arr

    def payload_as_columns(
        self, *, decode_enums: bool = True, demux_bit_masks: bool = False
    ) -> list[Column]:
        """Returns a list of Column where each member represents a field from a payload across multiple messages.

        ``decode_enums`` controls whether ``GroupMask`` enum columns become
        category codes and labels when ``True``, or raw integer codes when
        ``False``, which is a shape-preserving relabel. ``demux_bit_masks``
        controls whether a ``BitMask`` flag column is expanded into one boolean
        column per flag member when ``True``, or kept as a single raw-integer
        column when ``False``, which is a shape change. The two are orthogonal
        and apply to different descriptor kinds.
        """
        arr = np.atleast_1d(self._arr)
        # Each descriptor renders its own column(s); resolve via the scalar twin.
        scalar_cls = type(self)._scalar_cls
        cols: list[Column] = []
        for f in self._repr_fields:
            desc = scalar_cls._mro_descriptor(f)
            assert desc is not None
            cols.extend(
                desc._columns(arr, f, decode_enums=decode_enums, demux_bit_masks=demux_bit_masks)
            )
        return cols

    def __len__(self) -> int:
        return 1 if self._arr.ndim == 0 else len(self._arr)

    def _repr_kwargs(self) -> str:
        return ", ".join(f"{f}={getattr(self, f)!r}" for f in self._repr_fields)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._repr_kwargs()})"

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def _unwrap(cls, arr: "np.ndarray") -> Any:
        """Dispatch hook used by ``RegisterBase.parse``.

        Struct payloads always return a typed wrapper so descriptors like
        ``payload.Channel0`` work. Anonymous payloads override this to return the
        raw numpy scalar or ndarray directly, or, for an ``AnonymousPayload`` root,
        the unwrapped ``__value__`` through the single-member branch below, entered
        via the ``super()`` call of the override. A struct payload never auto-unwraps.
        """
        obj = cls._from_array(arr)
        if cls._single_member is not None and arr.ndim == 0:
            return getattr(obj, cls._single_member)
        return obj


# ---------------------------------------------------------------------------
# StructPayload, the base for named-field struct register payloads
# ---------------------------------------------------------------------------


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(Field, GroupMask, BitMask),
)
class StructPayload(PayloadBase[NpStructT]):
    """Base class for struct register payloads with typed field descriptors.

    Subclasses declare fields using ``Field``, ``GroupMask``, or ``BitMask``
    descriptors.  Type checkers synthesize a keyword-only ``__init__`` from
    those declarations, so constructor calls are fully type-checked and have
    IDE autocompletion.

    The type argument (``StructPayload[np.uint8]``) is the base element type; it
    sets the unit for ``offset=`` and the integer width of masked reads. The
    optional ``length=`` class kwarg fixes the payload size in base elements
    (the register ``length``); when omitted it defaults to the max member extent.

    Example::

        class MyPayload(StructPayload[np.uint8]):
            channel: np.uint16 = Field(UInt16Converter(), offset=0)
            flags: MyFlags = BitMask(enum=MyFlags, offset=2)
    """


# ---------------------------------------------------------------------------
# AnonymousPayload, a single unnamed slot with no user-facing descriptors
# ---------------------------------------------------------------------------


class AnonymousPayload(PayloadBase[NpStructT]):
    """Payload backed by a single unnamed numpy dtype (scalar or sub-array).

    Subclasses declare the dtype via class kwargs, not descriptors:

        class PayloadU16(AnonymousPayload, scalar_dtype="<u2"): ...

    Used for scalar/array Harp registers whose payload has no internal
    structure. ``RegisterBase.parse`` unwraps these to a raw numpy scalar
    for 0-D, or an ndarray for a sub-array or batch. There is no
    ``.value`` accessor and the slot name ``value`` is free for use by
    struct payloads.

    For a register that carries one *decoded* value (a codec, enum, or flag), a
    subclass declares a single ``__value__`` *descriptor* field.
    The descriptor may be a :class:`Field` (any :class:`Converter` codec),
    a :class:`GroupMask` (enum), or a :class:`BitMask`
    (flag)::

        class DeviceNamePayload(AnonymousPayload[np.uint8]):
            __value__: str = Field(StringConverter(25))

        class EncoderModePayload(AnonymousPayload[np.uint8]):
            __value__: EncoderModeMask = GroupMask(enum=EncoderModeMask, mask=0xFF)

        class ResetDevicePayload(AnonymousPayload[np.uint8]):
            __value__: ResetFlags = BitMask(enum=ResetFlags)

    The ``__value__`` field makes "this payload *is* one value" structural and
    explicit: exactly one field, named ``__value__``, unwrapped on ``parse`` and
    accessed as ``.__value__``. Declaring any other field is an error. Because the
    value stays a descriptor, ``to_columns`` keeps full rendering, meaning enum
    categoricals and ``demux_bit_masks`` flag expansion. ``__value__`` is mutually
    exclusive with ``scalar_dtype=``.

    Every concrete subclass must define its single slot exactly one way:
    ``__value__`` (a descriptor) or ``scalar_dtype=`` (a raw numpy dtype; an
    explicit ``dtype`` in the body, used by the array-register metaclass, also
    counts). Defining none is a definition-time error.
    """

    # The reserved field name for the single root view, as with __root__ in pydantic.
    _VALUE_FIELD: ClassVar[str] = "__value__"
    # True when in descriptor-root mode (the single view is the ``__value__`` field).
    _root: ClassVar[bool] = False

    def __init_subclass__(
        cls,
        *,
        scalar_dtype: "np.dtype | str | type | None" = None,
        **kwargs: object,
    ) -> None:
        # A class-body descriptor field selects root mode; it must be named __value__.
        body_fields = [
            n for n, v in cls.__dict__.items() if isinstance(v, _SCALAR_DECLARATION_TYPES)
        ]
        if body_fields:
            if scalar_dtype is not None:
                raise TypeError(
                    f"{cls.__name__}: __value__ is mutually exclusive with scalar_dtype="
                )
            if body_fields != [cls._VALUE_FIELD]:
                raise TypeError(
                    f"{cls.__name__}: a single-value (root) payload declares exactly one field "
                    f"named {cls._VALUE_FIELD!r}; found {body_fields}. Use StructPayload for "
                    f"multi-field payloads."
                )
            cls._root = True
            super().__init_subclass__(**kwargs)  # pyright: ignore[reportArgumentType]
            return
        # Raw scalar slot required, unless a Batch twin / array concrete supplies dtype.
        if (
            scalar_dtype is None
            and "_batch_of" not in kwargs
            and "payload_dtype" not in cls.__dict__
        ):
            raise TypeError(
                f"{cls.__name__}: an AnonymousPayload subclass must define its single slot via a "
                f"{cls._VALUE_FIELD!r} descriptor field or scalar_dtype= (a codec is a "
                f"{cls._VALUE_FIELD!r} Field with a Converter)."
            )
        if scalar_dtype is not None:
            cls.payload_dtype = np.dtype(scalar_dtype)
            cls._repr_fields = ()
        super().__init_subclass__(**kwargs)  # pyright: ignore[reportArgumentType]

    def __init__(self, value: object = _MISSING_INIT, /, **kwargs: object) -> None:  # type: ignore[override]
        if type(self)._root:
            # root mode: PayloadBase encodes the value into the single __value__ field
            if value is not _MISSING_INIT:
                super().__init__(value)
            else:
                super().__init__(**kwargs)
            return
        if value is _MISSING_INIT:
            if "value" in kwargs:
                value = kwargs.pop("value")
            else:
                raise TypeError(f"{type(self).__name__}() requires a value")
        if kwargs:
            raise TypeError(f"{type(self).__name__}() got unexpected kwargs: {sorted(kwargs)}")
        subdtype = self.payload_dtype.subdtype
        if subdtype is not None:
            element_dtype, shape = subdtype
            arr = np.asarray(value, dtype=element_dtype)
            if arr.shape != shape:
                expected = shape[0] if len(shape) == 1 else shape
                raise ValueError(
                    f"{type(self).__name__}() expects {expected} elements but got {arr.size}"
                )
        else:
            arr = np.asarray(value, dtype=self.payload_dtype)
        self._arr = arr

    @classmethod
    def _unwrap(cls, arr: "np.ndarray") -> Any:
        if cls._root:
            return super()._unwrap(arr)  # PayloadBase single-member unwrap (.__value__)
        # 0-D becomes a numpy scalar via item-like access, preserving dtype.
        # 1-D or sub-array returns the ndarray as-is.
        return arr if arr.ndim > 0 else arr[()]

    def _repr_kwargs(self) -> str:
        if type(self)._root:
            return super()._repr_kwargs()
        return repr(self._arr.tolist() if self._arr.ndim > 0 else self._arr[()])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._repr_kwargs()})"

    def payload_as_columns(
        self, *, decode_enums: bool = True, demux_bit_masks: bool = False
    ) -> list[Column]:
        # Anonymous values carry no name (name=None); the consumer supplies the label.
        if type(self)._root:
            arr = np.atleast_1d(self._arr)
            root = type(self)._scalar_cls._mro_descriptor(self._VALUE_FIELD)
            assert root is not None
            return root._columns(
                arr, None, decode_enums=decode_enums, demux_bit_masks=demux_bit_masks
            )
        arr = np.atleast_1d(self._arr)
        # Sub-array dtype (array register): one column per element, positionally named.
        if arr.ndim > 1:
            return [Column(str(i), arr[:, i]) for i in range(arr.shape[1])]
        return [Column(None, arr)]


@final
class PayloadU8(AnonymousPayload[np.uint8], scalar_dtype=np.dtype("u1")):
    pass


@final
class PayloadU16(AnonymousPayload[np.uint16], scalar_dtype=np.dtype("<u2")):
    pass


@final
class PayloadU32(AnonymousPayload[np.uint32], scalar_dtype=np.dtype("<u4")):
    pass


@final
class PayloadU64(AnonymousPayload[np.uint64], scalar_dtype=np.dtype("<u8")):
    pass


@final
class PayloadS8(AnonymousPayload[np.int8], scalar_dtype=np.dtype("i1")):
    pass


@final
class PayloadS16(AnonymousPayload[np.int16], scalar_dtype=np.dtype("<i2")):
    pass


@final
class PayloadS32(AnonymousPayload[np.int32], scalar_dtype=np.dtype("<i4")):
    pass


@final
class PayloadS64(AnonymousPayload[np.int64], scalar_dtype=np.dtype("<i8")):
    pass


@final
class PayloadFloat(AnonymousPayload[np.float32], scalar_dtype=np.dtype("<f4")):
    pass


# Array payload base classes, with the concrete sub-dtype set by ``RegisterBase``
# array metaclass when a length is supplied (e.g. ``RegisterU16Array(0x28, length=3)``).
@final
class PayloadU8Array(AnonymousPayload[np.uint8], scalar_dtype=np.dtype("u1")):
    pass


@final
class PayloadU16Array(AnonymousPayload[np.uint16], scalar_dtype=np.dtype("<u2")):
    pass


@final
class PayloadU32Array(AnonymousPayload[np.uint32], scalar_dtype=np.dtype("<u4")):
    pass


@final
class PayloadU64Array(AnonymousPayload[np.uint64], scalar_dtype=np.dtype("<u8")):
    pass


@final
class PayloadS8Array(AnonymousPayload[np.int8], scalar_dtype=np.dtype("i1")):
    pass


@final
class PayloadS16Array(AnonymousPayload[np.int16], scalar_dtype=np.dtype("<i2")):
    pass


@final
class PayloadS32Array(AnonymousPayload[np.int32], scalar_dtype=np.dtype("<i4")):
    pass


@final
class PayloadS64Array(AnonymousPayload[np.int64], scalar_dtype=np.dtype("<i8")):
    pass


@final
class PayloadFloatArray(AnonymousPayload[np.float32], scalar_dtype=np.dtype("<f4")):
    pass
