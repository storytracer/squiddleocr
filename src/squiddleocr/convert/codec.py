"""Convert a kraken codec (grapheme -> labels) into a PaddleOCR character dictionary."""
from __future__ import annotations

from pathlib import Path


def codec_to_dictionary(c2l: dict[str, list[int]]) -> list[str]:
    """Turn kraken's ``c2l`` mapping into an ordered PaddleOCR dictionary.

    The returned list has one entry per label in label order: entry ``i``
    corresponds to CTC class ``i + 1``. Class 0 (the CTC blank) is not part of
    the list; PaddleX's ``CTCLabelDecode`` prepends ``blank`` itself.

    Requirements, all checked:

    * every grapheme maps to exactly one label,
    * labels are ``1..N`` without gaps or duplicates,
    * no grapheme contains a line break (the dictionary is also written one
      entry per line).
    """
    if not c2l:
        raise ValueError("Empty codec.")
    label_to_char: dict[int, str] = {}
    for char, labels in c2l.items():
        if len(labels) != 1:
            raise ValueError(
                f"Grapheme {char!r} maps to {len(labels)} labels; PaddleOCR CTC dictionaries "
                "need a one-to-one mapping."
            )
        label = int(labels[0])
        if label < 1:
            raise ValueError(f"Grapheme {char!r} has label {label}; 0 is reserved for the CTC blank.")
        if label in label_to_char:
            raise ValueError(
                f"Label {label} is used by both {label_to_char[label]!r} and {char!r}."
            )
        if "\n" in char or "\r" in char:
            raise ValueError(f"Grapheme {char!r} contains a line break.")
        if char == "":
            raise ValueError("Empty grapheme in codec.")
        label_to_char[label] = char
    expected = set(range(1, len(label_to_char) + 1))
    if set(label_to_char) != expected:
        missing = sorted(expected - set(label_to_char))[:5]
        extra = sorted(set(label_to_char) - expected)[:5]
        raise ValueError(
            f"Labels must be contiguous 1..{len(label_to_char)}; missing {missing}, unexpected {extra}."
        )
    return [label_to_char[i] for i in range(1, len(label_to_char) + 1)]


def write_dict_file(chars: list[str], path: str | Path) -> None:
    """Write a PaddleOCR-style dictionary file: one entry per line, UTF-8, no trailing whitespace stripping.

    A lone space entry is written as a line containing a single space; PaddleOCR's
    file reader strips only line terminators, so it survives.
    """
    Path(path).write_text("".join(c + "\n" for c in chars), encoding="utf-8")


def read_dict_file(path: str | Path) -> list[str]:
    """Read a dictionary file written by :func:`write_dict_file`."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return [ln.rstrip("\r") for ln in lines]

