"""Track data containers for the INTERACTION dataset.

This module provides the following containers:

- :class:`MotionState`: A single motion state of an agent at a time step.
- :class:`Track`: A single track.
- :class:`INTERACTIONCase`: A single case of observation in a scenario.

The containers are designed to be immutable.
"""
# Copyright (c) 2023, Juanwu Lu <juanwu@purdue.edu>.
# Released under the BSD-3-Clause license.
# See https://opensource.org/license/bsd-3-clause/ for licensing details.
import math
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon, Point

from .typing import AgentType


@dataclass(frozen=True)
class MotionState:
    """A single motion state of an agent at a time step."""

    agent_id: int
    """int: The ID of the agent in a case."""
    timestamp_ms: int
    """int: The timestamp of the motion state in miliseconds."""
    position_x: float
    """float: The x-coordinate of the position of the agent in meters."""
    position_y: float
    """float: The y-coordinate of the position of the agent in meters."""
    velocity_x: float
    """float: The x-component of the velocity in meters per second."""
    velocity_y: float
    """float: The y-component of the velocity in meters per second."""
    heading: Optional[float] = None
    """float: The heading of the agent in radians."""
    length: Optional[float] = None
    """float: The length of the agent in meters."""
    width: Optional[float] = None
    """width: The width of the agent in meters."""

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        assert (
            isinstance(self.agent_id, int) and self.agent_id >= 0
        ), f"Agent ID must be a non-negative int, but got {self.agent_id}"
        assert (
            isinstance(self.timestamp_ms, int) and self.timestamp_ms >= 0.0
        ), f"Timestamp must be a non-negative int, but got {self.timestamp_ms}"
        assert isinstance(self.position_x, float), "Position x must be a float"
        assert isinstance(self.position_y, float), "Position y must be a float"
        assert isinstance(self.heading, float), "Heading must be a float"
        assert isinstance(self.velocity_x, float), "Velocity x must be a float"
        assert isinstance(self.velocity_y, float), "Velocity y must be a float"
        assert isinstance(self.length, float), "Length must be a float"
        assert isinstance(self.width, float), "Width must be a float"

    @cached_property
    def bounding_box(self) -> Optional[Polygon]:
        """Optional[Polygon]: The bounding box of the current motion state."""
        if self.heading is None or self.length is None or self.width is None:
            return None

        half_length = self.length / 2.0
        half_width = self.width / 2.0

        corners = np.array(
            [
                # rear-right corner
                (-half_length, -half_width),
                # rear-left corner
                (-half_length, half_width),
                # front-left corner
                (half_length, half_width),
                # front-right corner
                (half_length, -half_width),
            ],
            dtype=np.float64,
        )

        rotation_matrix = np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading)],
                [np.sin(self.heading), np.cos(self.heading)],
            ],
            dtype=np.float64,
        )
        translation_vector = np.array([self.x, self.y], dtype=np.float64)
        corners = corners @ rotation_matrix.T + translation_vector

        return Polygon(corners)

    @property
    def speed(self) -> float:
        """float: The speed of the agent in meters per second."""
        return math.hypot(self.velocity_x, self.velocity_y)

    def to_geometry(self) -> Point:
        """Convert the motion state to a Shapely geometry object.

        Returns:
            Point: The motion state as a Shapely geometry object.
        """
        return Point(self.position_x, self.position_y)

    def __eq__(self, __value: Any) -> bool:
        if isinstance(__value, MotionState):
            return hash(self) == hash(__value)
        return NotImplemented

    def __ne__(self, __value: Any) -> bool:
        if isinstance(__value, MotionState):
            return hash(self) != hash(__value)
        return NotImplemented

    def __ge__(self, __value: Any) -> bool:
        if (
            isinstance(__value, MotionState)
            and self.agent_id == __value.agent_id
        ):
            return self.timestamp_ms >= __value.timestamp_ms
        return NotImplemented

    def __gt__(self, __value: Any) -> bool:
        if (
            isinstance(__value, MotionState)
            and self.agent_id == __value.agent_id
        ):
            return self.timestamp_ms > __value.timestamp_ms
        return NotImplemented

    def __le__(self, __value: Any) -> bool:
        if (
            isinstance(__value, MotionState)
            and self.agent_id == __value.agent_id
        ):
            return self.timestamp_ms <= __value.timestamp_ms
        return NotImplemented

    def __lt__(self, __value: Any) -> bool:
        if (
            isinstance(__value, MotionState)
            and self.agent_id == __value.agent_id
        ):
            return self.timestamp_ms < __value.timestamp_ms
        return NotImplemented

    def __hash__(self) -> int:
        return hash(
            (
                self.agent_id,
                self.timestamp_ms,
                self.position_x,
                self.position_y,
                self.velocity_x,
                self.velocity_y,
                self.heading,
                self.length,
                self.width,
            )
        )

    def __str__(self) -> str:
        attr_str = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"<{self.__class__.__name__}({attr_str}) at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class Track:
    """A single track consiting of multiple motion states of the same agent."""

    agent_id: int
    """int: The ID of the agent in a case."""
    type: AgentType
    """AgentType: The type of the agent."""
    motion_states: Tuple[MotionState, ...] = ()
    """Tuple[MotionState, ...]: The motion states of the agent."""

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        assert (
            isinstance(self.agent_id, int) and self.agent_id >= 0
        ), "Agent ID must be a non-negative int"
        assert isinstance(
            self.type, AgentType
        ), "Agent type must be an `AgentType` object."
        assert isinstance(self.motion_states, Iterable) and all(
            isinstance(ms, MotionState) and ms.agent_id == self.agent_id
            for ms in self.motion_states
        ), "Motion states must be an iterable of `MotionState` objects with the same agent ID."

        # sort the motion states by timestamp
        self.motion_states = tuple(sorted(self.motion_states))

    @cached_property
    def bounding_boxes(self) -> List[Polygon]:
        """List[Polygon]: The bounding boxes of the track."""
        return [
            ms.bounding_box for ms in self.motion_states if ms.bounding_box
        ]

    @cached_property
    def timestamps(self) -> List[int]:
        """List[int]: The timestamps of the track in milliseconds."""
        return [ms.timestamp_ms for ms in self.motion_states]

    @cached_property
    def min_timestamp_ms(self) -> float:
        """float: The minimum timestamp of the track in milliseconds."""
        return min(ms.timestamp_ms for ms in self.motion_states)

    @cached_property
    def max_timestamp_ms(self) -> float:
        """float: The maximum timestamp of the track in milliseconds."""
        return max(ms.timestamp_ms for ms in self.motion_states)

    @property
    def num_motion_states(self) -> int:
        """int: The number of motion states in the track."""
        return len(self.motion_states)

    def to_geometry(self) -> LineString:
        """Convert the track to a Shapely geometry object.

        Returns:
            LineString: The track as a Shapely geometry object.
        """
        return LineString(
            [(ms.position_x, ms.position_y) for ms in self.motion_states]
        )

    def __getitem__(self, __index: int) -> MotionState:
        """Get the motion state at the given index."""
        return self.motion_states[__index]

    def __iter__(self) -> Iterator[MotionState]:
        """Return an iterator over the motion states of the track."""
        return iter(self.motion_states)

    def __len__(self) -> int:
        """Get the number of motion states in the track."""
        return len(self.motion_states)

    def __hash__(self) -> int:
        return hash((self.agent_id, self.type, self.motion_states))

    def __str__(self) -> str:
        attr_str = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"<{self.__class__.__name__}({attr_str}) at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class INTERACTIONCase:
    """A single case of observation in a scenario."""

    location: str
    """str: The location name of the case."""
    case_id: int
    """int: The ID of the case."""
    history_tracks: Tuple[Track] = ()
    """Tuple[Track, ...]: The history tracks of the case."""
    current_tracks: Tuple[Track] = ()
    """Tuple[Track, ...]: The current tracks of the case."""
    futural_tracks: Tuple[Track] = ()
    """Tuple[Track, ...]: The futural tracks of the case."""
    tracks_to_predict: Tuple[int] = ()
    """Tuple[int, ...]: The IDs of the tracks to predict."""
    interesting_agents: Tuple[int] = ()
    """Tuple[int, ...]: The IDs of the interesting agents (ego vehicles)."""

    @cached_property
    def num_agents(self) -> int:
        """int: The number of agents in the case."""
        return len(
            set(
                track.agent_id
                for track in chain(
                    self.history_tracks,
                    self.current_tracks,
                    self.futural_tracks,
                )
            )
        )

    def __eq__(self, __value: Any) -> bool:
        if isinstance(__value, INTERACTIONCase):
            return (
                self.location == __value.location
                and self.case_id == __value.case_id
            )
        return NotImplemented

    def __ne__(self, __value: Any) -> bool:
        if isinstance(__value, INTERACTIONCase):
            return (
                self.location != __value.location
                or self.case_id != __value.case_id
            )
        return NotImplemented

    def __ge__(self, __value: Any) -> bool:
        if (
            isinstance(__value, INTERACTIONCase)
            and self.location == __value.location
        ):
            return self.case_id >= __value.case_id
        return NotImplemented

    def __gt__(self, __value: Any) -> bool:
        if (
            isinstance(__value, INTERACTIONCase)
            and self.location == __value.location
        ):
            return self.case_id > __value.case_id
        return NotImplemented

    def __le__(self, __value: Any) -> bool:
        if (
            isinstance(__value, INTERACTIONCase)
            and self.location == __value.location
        ):
            return self.case_id <= __value.case_id
        return NotImplemented

    def __lt__(self, __value: Any) -> bool:
        if (
            isinstance(__value, INTERACTIONCase)
            and self.location == __value.location
        ):
            return self.case_id < __value.case_id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(
            (
                self.location,
                self.case_id,
                self.history_tracks,
                self.current_tracks,
                self.futural_tracks,
                self.tracks_to_predict,
                self.interesting_agents,
            )
        )

    def __str__(self) -> str:
        attr_str = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"<{self.__class__.__name__}({attr_str}) at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)
