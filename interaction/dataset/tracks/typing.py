"""Typing for the track dataset.

This module contains the type definitions for the track dataset elements. Each
type definition is an enumeration of the valid values for the corresponding
element type. The values are categorical and can be serialized to one-hot
vectors using the `one_hot_serialize` method:

    >>> from interaction_devkit.dataset.tracks.typing import TrackType
    >>> assert TrackType.CAR.one_hot_serialize() == [1, 0]
"""
# Copyright (c) 2023, Juanwu Lu <juanwu@purdue.edu>.
# Released under the BSD-3-Clause license.
# See https://opensource.org/license/bsd-3-clause/ for licensing details.
from enum import IntEnum
from typing import Union


class AgentType(IntEnum):
    """Enumeration of the valid types of `agent` elements."""

    UNDEFINED = 0
    """The placeholder value for undefined `agent` elements."""
    CAR = 1
    """The observed `agent` is a car."""
    PEDESTRIAN_BICYCLE = 2
    """The observed `agent` is a pedestrian or a bicycle."""

    @classmethod
    def deserialize(cls, value: Union[int, str]) -> "AgentType":
        """Deserialize an `AgentType` from a string.

        Args:
            value (Union[int, str]): the value to deserialize.

        Returns:
            AgentType: the deserialized `AgentType` value.

        Raises:
            TypeError: if the value cannot be deserialized.
        """
        if isinstance(value, str):
            return cls[value.strip().replace("/", "_").upper()]
        elif isinstance(value, int):
            return cls(value)
        else:
            raise TypeError(f"Unable to deserialize {value} to AgentType.")

    def one_hot_serialize(self) -> list[int]:
        ret = [0 for _ in range(max(AgentType))]
        if self.value > 0:
            ret[self.value - 1] = 1
        return ret
