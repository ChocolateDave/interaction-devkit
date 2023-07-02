"""Speed limit regulation of the lanelet."""
# Copyright (c) 2023, Juanwu Lu <juanwu@purdue.edu>.
# Released under the BSD-3-Clause license.
# See https://opensource.org/license/bsd-3-clause/ for licensing details.
import re
from typing import Any, Callable


class SpeedLimit:
    """Speed limit regulation of the lanelet."""

    valid_units: tuple[str] = (
        "mps",
        "kmph",
        "kmh",
        "kph",
        "km/h",
        "mph",
        "mi/h",
    )
    """Valid units for speed limits."""

    __slots__ = "_speed_limit_mps"

    def __init__(self, speed_limit_mps: float, _direct: bool = True) -> None:
        """Construct a speed limit regulation.

        Args:
            speed_limit_mps (float): The speed limit in meters per second.
            _direct (bool): Whether the constructor is called directly or from
                a factory method.
        """
        if _direct:
            raise RuntimeError(
                "SpeedLimit must be created with from_* methods"
            )
        self._speed_limit_mps = speed_limit_mps

    @classmethod
    def from_mps(cls, speed_limit_mps: float) -> "SpeedLimit":
        """Construct a speed limit regulation from meters per second.

        Args:
            speed_limit_mps (float): The speed limit in meters per second.

        Returns:
            SpeedLimit: A speed limit regulation.
        """
        return cls(speed_limit_mps, _direct=False)

    @classmethod
    def from_kmph(cls, speed_limit_kmph: float) -> "SpeedLimit":
        """Construct a speed limit regulation from kilometers per hour.

        Args:
            speed_limit_kmph (float): The speed limit in kilometers per hour.

        Returns:
            SpeedLimit: A speed limit regulation.
        """
        return cls(speed_limit_kmph / 3.6, _direct=False)

    @classmethod
    def from_mph(cls, speed_limit_mph: float) -> "SpeedLimit":
        """Construct a speed limit regulation from miles per hour.

        Args:
            speed_limit_mph (float): The speed limit in miles per hour.

        Returns:
            A speed limit regulation.
        """
        return cls(speed_limit_mph * 0.44704, _direct=False)

    @classmethod
    def from_string(cls, speed_limit_string: str) -> "SpeedLimit":
        """Construct a speed limit regulation from a string.

        Args:
            speed_limit_string (str): A string containing a speed limit. The
                speed limit must be followed by a unit, e.g. "50mph".

        Returns:
            SpeedLimit: A speed limit regulation.
        """
        value, unit = None, None
        for unit in cls.valid_units:
            match = re.search(rf"{unit}", speed_limit_string)
            if match:
                value = float(speed_limit_string[: match.start()].strip())
                break
        assert value is not None, "Invalid speed limit string"

        return _SPEED_LIMIT_FACTORY[unit](float(value))

    @property
    def speed_limit_mps(self) -> float:
        """Return the speed limit in meters per second."""
        return self._speed_limit_mps

    @property
    def speed_limit_kmph(self) -> float:
        """Return the speed limit in kilometers per hour."""
        return self._speed_limit_mps * 3.6

    @property
    def speed_limit_mph(self) -> float:
        """Return the speed limit in miles per hour."""
        return self._speed_limit_mps / 0.44704

    def __hash__(self) -> int:
        """Return the hash of the speed limit."""
        return hash(self._speed_limit_mps)

    def __str__(self) -> str:
        """Return the speed limit as a string."""
        return (
            f"<{self.__class__.__name__}({self.speed_limit_mps:.1f} mps)"
            f"at {hex(id(self))}>"
        )

    def __repr__(self) -> str:
        """Return the string representation of the speed limit."""
        return str(self)


_SPEED_LIMIT_FACTORY: dict[
    str,
    Callable[
        [
            Any,
        ],
        "SpeedLimit",
    ],
] = {
    "mps": SpeedLimit.from_mps,
    "kmph": SpeedLimit.from_kmph,
    "kmh": SpeedLimit.from_kmph,
    "kph": SpeedLimit.from_kmph,
    "km/h": SpeedLimit.from_kmph,
    "mph": SpeedLimit.from_mph,
    "mi/h": SpeedLimit.from_mph,
}
