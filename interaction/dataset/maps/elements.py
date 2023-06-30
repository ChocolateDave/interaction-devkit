"""Lanelet2 map element data containers.

Copyright (c) 2023, Juanwu Lu. Released under the BSD-3-Clause license.
"""
import abc
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString, Point, Polygon

from .typing import (
    LaneletSubType,
    WayType,
    MultiPolygonSubType,
    RegulatoryElementSubType,
)
from .speed_limit import SpeedLimit


@dataclass
class MapElement(object):
    """Base class for map elements."""

    id: int
    """Unique identifier of the map element."""

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        self.id = int(self.id)
        assert (
            isinstance(self.id, int) and self.id >= 0
        ), f"Map element ID must be a non-negative int, but got {self.id}"

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "MapElement":
        """Deserialize a map element from a dictionary."""
        attrs: Dict[str, Any] = {}
        for field_ in fields(cls):
            assert field_.name in data, f"Missing field {field_.name}"
            attrs[field_.name] = data[field_.name]

        return cls(**data)

    @property
    def field_names(self) -> List[str]:
        """Return the field names of the map element."""
        return [field_.name for field_ in fields(self)]

    def serialize(self) -> Dict[str, Any]:
        """Serialize a map element to a dictionary."""
        return {
            field_.name: getattr(self, field_.name) for field_ in fields(self)
        }

    @abc.abstractmethod
    def to_geometry(self) -> BaseGeometry:
        """Convert the map element to a Shapely geometry object."""
        raise NotImplementedError

    def __hash__(self) -> int:
        """Return hash of the map element."""
        return hash(self.id)

    def __str__(self):
        """Return string representation of the map element."""
        return f"<{self.__class__.__name__}({self.id}) at {hex(id(self))}>"

    def __repr__(self):
        """Return string representation of the map element."""
        return str(self)


@dataclass
class Node(MapElement):
    """Data class representing a node elemenet in the map data."""

    x: float
    """X coordinate of the node in meters."""
    y: float
    """Y coordinate of the node in meters."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert isinstance(self.x, float), "Node x coordinate must be a float"
        assert isinstance(self.y, float), "Node y coordinate must be a float"

    def to_geometry(self) -> Point:
        return Point(self.x, self.y)

    def __hash__(self) -> int:
        return hash((self.id, self.x, self.y))


@dataclass
class Way(MapElement):
    """Data class representing a way element in the map data."""

    type: WayType
    """Type of the way."""
    nodes: Tuple[Node]
    """Tuple of :obj:`Node` objects that make up the way."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert isinstance(self.type, WayType), "Way type must be a `WayType`."
        assert isinstance(self.nodes, Iterable), "Way nodes must be iterable."
        self.nodes = tuple(self.nodes)

    def to_geometry(self) -> LineString:
        return LineString([(node.x, node.y) for node in self.nodes])

    @property
    def node_ids(self) -> Tuple[int]:
        """The IDs of the nodes that make up the way."""
        return tuple(node.id for node in self.nodes)

    def __hash__(self) -> int:
        return hash((self.id, self.type, self.nodes))


@dataclass
class Lanelet(MapElement):
    """Data class representing a lanelet element in the map data."""

    subtype: LaneletSubType
    """Subtype of the lanelet."""
    left: Way
    """Left boundary of the lanelet."""
    right: Way
    """Right boundary of the lanelet."""
    speed_limit: SpeedLimit
    """Speed limit regulation of the lanelet."""
    stop_line: Optional[Way] = None
    """Stop line of the lanelet."""
    adjacent_lanelets: Tuple["Lanelet"] = ()
    """Adjacent (left/right) lanelets of the current lanelet."""
    predecessor_lanelets: Tuple["Lanelet"] = ()
    """Predecessor (i.e., upstream inflow) lanelets of the current lanelet."""
    successor_lanelets: Tuple["Lanelet"] = ()
    """Successor (i.e., downstream outflow) lanelets of the current lanelet."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert isinstance(
            self.subtype, LaneletSubType
        ), "Lanelet subtype must be a `LaneletSubType`."
        assert isinstance(
            self.left, Way
        ), "Lanelet left boundary must be a `Way`."
        assert isinstance(
            self.right, Way
        ), "Lanelet right boundary must be a `Way`."
        assert isinstance(
            self.speed_limit, SpeedLimit
        ), "Lanelet speed limit must be a `SpeedLimit`."
        assert isinstance(
            self.stop_line, (Way, type(None))
        ), "Lanelet stop line must be a `Way` or `None`."
        assert all(
            isinstance(lanelet, Lanelet) for lanelet in self.adjacent_lanelets
        ), "Lanelet adjacent lanelets must be `Lanelet` objects."
        assert all(
            isinstance(lanelet, Lanelet)
            for lanelet in self.predecessor_lanelets
        ), "Lanelet predecessor lanelets must be `Lanelet` objects."
        assert all(
            isinstance(lanelet, Lanelet) for lanelet in self.successor_lanelets
        ), "Lanelet successor lanelets must be `Lanelet` objects."

    def to_geometry(self) -> Polygon:
        return Polygon(
            [
                *[(node.x, node.y) for node in self.right.nodes],
                *[(node.x, node.y) for node in reversed(self.left.nodes)],
            ]
        )

    def __hash__(self) -> int:
        return hash(
            self.id, self.subtype, self.left, self.right, self.speed_limit
        )


@dataclass
class MultiPolygon(MapElement):
    """Data class representing a multipolygon element in the map data."""

    subtype: MultiPolygonSubType
    """Subtype of the multipolygon."""
    outer: Tuple[Way]
    """A tuple of ways that make up the outer boundary of the multipolygon."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert isinstance(
            self.subtype, MultiPolygonSubType
        ), "MultiPolygon subtype must be a `MultiPolygonSubType`."
        assert all(
            isinstance(way, Way) for way in self.outer
        ), "MultiPolygon outer ways must be `Way` objects."

    def to_geometry(self) -> Polygon:
        return Polygon(
            [
                *[
                    (node.x, node.y)
                    for outer_way in self.outer
                    for node in outer_way.nodes
                ],
            ]
        )

    def __hash__(self) -> int:
        return hash(self.id, self.subtype, self.outer)


@dataclass
class RegulatoryElement(MapElement):
    """Data class representing a regulatory element in the map data."""

    subtype: RegulatoryElementSubType
    """Subtype of the regulatory element."""
    refers: Tuple[Way] = ()
    """:obj:`Way` object representing the entity of the regulatory element."""
    ref_lines: Tuple[Way] = ()
    """:obj:`Way` objects representing the referencing lines."""
    prior_lanelets: Tuple[Lanelet] = ()
    """Lanelets that have right-of-way under the regulatory element."""
    yield_lanelets: Tuple[Lanelet] = ()
    """Lanelets that have to yield under the regulatory element."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert isinstance(
            self.subtype, RegulatoryElementSubType
        ), "RegulatoryElement subtype must be a `RegulatoryElementSubType`."
        assert all(
            isinstance(way, Way) and way.type == WayType.TRAFFIC_SIGN
            for way in self.refers
        ), "RegulatoryElement refers must be `Way` of `TRAFFIC_SIGN` type."
        assert all(
            isinstance(way, Way) for way in self.ref_lines
        ), "RegulatoryElement ref_lines must be `Way` objects."
        assert all(
            isinstance(lanelet, Lanelet) for lanelet in self.prior_lanelets
        ), "RegulatoryElement prior_lanelets must be `Lanelet` objects."
        assert all(
            isinstance(lanelet, Lanelet) for lanelet in self.yield_lanelets
        ), "RegulatoryElement yield_lanelets must be `Lanelet` objects."

    def to_geometry(self) -> List[LineString]:
        return [
            LineString([(node.x, node.y) for node in way.nodes])
            for way in self.refers
        ]

    def __hash__(self) -> int:
        return hash(
            self.id,
            self.subtype,
            self.refer,
            self.ref_lines,
            self.prior_lanelets,
            self.yield_lanelets,
        )
