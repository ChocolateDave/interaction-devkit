"""Typing for the map dataset.

This module contains the type definitions for the map dataset elements. Each
type definition is an enumeration of the valid values for the corresponding
element type. The values are categorical and can be serialized to one-hot
vectors using the `one_hot_serialize` method:

    >>> from interaction_devkit.dataset.maps.typing import WayType
    >>> assert WayType.CURBSTONE_LOW.one_hot_serialize() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"""
# Copyright (c) 2023, Juanwu Lu <juanwu@purdue.edu>.
# Released under the BSD-3-Clause license.
# See https://opensource.org/license/bsd-3-clause/ for licensing details.
from enum import IntEnum


class WayType(IntEnum):
    """Enumeration of the valid types of `way` elements."""

    UNDEFINED = 0
    """The placeholder value for undefined `way` elements."""
    CURBSTONE_LOW = 1
    """The `way` element is a low curbstone."""
    GUARD_RAIL = 2
    """The `way` element is a guard rail."""
    ROAD_BORDER = 3
    """The `way` element is a road border."""
    LINE_THIN_SOLID = 4
    """The `way` element is a thin solid lane line."""
    LINE_THIN_SOLID_SOLID = 4
    """The `way` element is a thin solid lane line."""
    LINE_THIN_DASHED = 5
    """The `way` element is a thin dashed lane line."""
    LINE_THICK_SOLID = 6
    """The `way` element is a thick solid lane line."""
    LINE_THICK_SOLID_SOLID = 6
    """The `way` element is a thick solid lane line."""
    STOP_LINE = 7
    """The `way` element is a stop line."""
    PEDESTRIAN_MARKING = 8
    """The `way` element is a pedestrian crossing marking."""
    VIRTUAL = 9
    """The `way `element is a virtual lane line."""
    VIRTUAL_SOLID = 9
    """The `way `element is a solid virtual lane line."""
    TRAFFIC_SIGN = 10
    """The `way` element is a traffic sign."""

    def one_hot_serialize(self) -> list[int]:
        ret = [0 for _ in range(max(WayType))]
        if self.value > 0:
            ret[self.value - 1] = 1
        return ret


class LaneletSubType(IntEnum):
    """Enumeration of the valid subtypes of `lanelet` elements."""

    UNDEFINED = 0
    """The placeholder value for undefined `lanelet` elements."""
    ROAD = 1
    """The `lanelet` element is a urban road lane."""
    HIGHWAY = 2
    """The `lanelet` element is a highway lane."""

    def one_hot_serialize(self) -> list[int]:
        ret = [0 for _ in range(max(LaneletSubType))]
        if self.value > 0:
            ret[self.value - 1] = 1
        return ret


class MultiPolygonSubType(IntEnum):
    """Enumeration of the valid subtypes of `multipolygon` elements."""

    UNDEFINED = 0
    """The placeholder value for undefined `multipolygon` elements."""
    KEEPOUT = 1
    """The `multipolygon` element is a keepout zone."""

    def one_hot_serialize(self) -> list[int]:
        ret = [0 for _ in range(max(MultiPolygonSubType))]
        if self.value > 0:
            ret[self.value - 1] = 1
        return ret


class RegulatoryElementSubType(IntEnum):
    """Enumeration of the valid subtypes of `regulatory_element` elements."""

    UNDEFINED = 0
    """The placeholder value for undefined `regulatory_element` elements."""
    SPEED_LIMIT = 1
    """The `regulatory_element` element is a speed limit sign."""
    RIGHT_OF_WAY = 2
    """The `regulatory_element` element is a stop sign enforce right-of-way."""
    ALL_WAY_STOP = 3
    """The `regulatory_element` element is a stop sign enforce all-way stop."""

    def one_hot_serialize(self) -> list[int]:
        ret = [0 for _ in range(max(RegulatoryElementSubType))]
        if self.value > 0:
            ret[self.value - 1] = 1
        return ret


class RightOfWay(IntEnum):
    """Enumeration of the valid right-of-way values."""

    UNDEFINED = 0
    """The placeholder value for undefined right-of-way."""
    YIELD = 1
    """The right-of-way is yielded."""
    RIGHT_OF_WAY = 2
    """The right-of-way is ensured."""

    def one_hot_serialize(self) -> list[int]:
        ret = [0 for _ in range(max(RightOfWay))]
        if self.value > 0:
            ret[self.value - 1] = 1
        return ret
