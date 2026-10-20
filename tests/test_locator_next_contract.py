"""``Locator.next()`` is a unit drive frequency in [0, 1]; ``next_drive_freq_unit()`` enforces it."""

from __future__ import annotations

import pytest

from nvision.models.locator import Locator


class _FixedNextLocator(Locator):
    def __init__(self, value: float) -> None:
        super().__init__(belief=None)  # type: ignore[arg-type]
        self._value = value

    @classmethod
    def create(cls, **config):  # pragma: no cover - unused
        raise NotImplementedError

    def next(self) -> float:
        return self._value

    def done(self) -> bool:  # pragma: no cover - unused
        return False

    def result(self) -> dict[str, float]:  # pragma: no cover - unused
        return {}


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_unit_drive_freq_is_returned_unchanged(value):
    assert _FixedNextLocator(value).next_drive_freq_unit() == value


@pytest.mark.parametrize("value", [2.87e9, -0.01, 1.0001, float("nan")])
def test_physical_or_out_of_range_next_raises(value):
    with pytest.raises(ValueError, match="unit drive frequency"):
        _FixedNextLocator(value).next_drive_freq_unit()
