"""Coordinate projector for the raw map dataset."""
# Copyright (c) 2023, Juanwu Lu <juanwu@purdue.edu>.
# Released under the BSD-3-Clause license.
# See https://opensource.org/license/bsd-3-clause/ for licensing details.
import math

import pyproj


class INTERACTIONProjector:
    """A wrapper of `pyproj.Proj` for projecting INTERACTION map element
    coordinates.

    Attributes:
        origin_x: The x coordinate of the origin.
        origin_y: The y coordinate of the origin.
    """

    __slots__ = "_origin_x", "_origin_y", "_proj"

    def __init__(
        self, origin_lon: float = 0.0, origin_lat: float = 0.0
    ) -> None:
        """Constructor of `INTERACTIONProjector`.

        Args:
            origin_lon: The longitude of the origin. Defaults to 0.0.
            origin_lat: The latitude of the origin. Defaults to 0.0.
        """
        super().__init__()
        _zone = math.floor((origin_lon + 180.0) / 6.0) + 1
        self._proj = pyproj.Proj(
            proj="utm", ellps="WGS84", zone=_zone, datum="WGS84"
        )
        self._origin_x, self._origin_y = self._proj(origin_lon, origin_lat)

    def __call__(
        self, lon: float, lat: float, relative: bool = True, *args, **kwargs
    ) -> tuple[float, float]:
        ret_x, ret_y = self._proj(lon, lat, inverse=False, *args, **kwargs)
        if relative:
            ret_x = ret_x - self._origin_x
            ret_y = ret_y - self._origin_y

        return ret_x, ret_y

    @property
    def origin_x(self) -> float:
        """float: The x coordinate of the coordinate origin."""
        return self._origin_x

    @origin_x.setter
    def origin_x(self, value: float) -> None:
        self._origin_x = float(value)

    @property
    def origin_y(self) -> float:
        """float: The y coordinate of the coordinate origin."""
        return self._origin_y

    @origin_y.setter
    def origin_y(self, value: float) -> None:
        self._origin_y = float(value)

    @property
    def origin(self) -> tuple[float, float]:
        """Tuple[float, float]: The coordinate origin."""
        return self._origin_x, self._origin_y

    def __repr__(self) -> str:
        attr_str = f"ori_x={self._origin_x}, ori_y={self._origin_y}"
        return f"<{self.__class__.__name__}({attr_str}) at {hex(id(self))}>"
