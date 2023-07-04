"""INTERACTION map data API.

This module defines the API for the INTERACTION map data. The raw map data is
stored in Lanelet2 map format `.osm` files. The map data API parses the `.osm`
files and provides a convenient interface for accessing the map data.
"""
# Copyright (c) 2023, Juanwu Lu <juanwu@purdue.edu>.
# Released under the BSD-3-Clause license.
# See https://opensource.org/license/bsd-3-clause/ for licensing details.
import math
import os
from collections import defaultdict
from collections.abc import Generator, Iterable
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Optional, Union
from xml import etree

import geopandas as gpd
import matplotlib.pyplot as plt
from defusedxml.ElementTree import parse as safe_parse
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.ops import substring

from .maps.elements import (
    Lanelet,
    MapElement,
    MultiPolygon,
    Node,
    RegulatoryElement,
    Way,
)
from .maps.projector import INTERACTIONProjector
from .maps.speed_limit import SpeedLimit
from .maps.typing import (
    LaneletSubType,
    MultiPolygonSubType,
    RegulatoryElementSubType,
    WayType,
)
from .maps.utils import (
    LANELET_TO_REVERSE,
    get_linestring_direction,
    get_way_type,
    instantiate_way,
    order_ways,
    reverse_way,
)

# Constants
WAY_STYLE_MAPPING: dict[WayType, dict[str, Any]] = {
    WayType.UNDEFINED: dict(alpha=0.0),
    WayType.CURBSTONE_LOW: dict(color="#919595", linewidth=1.5, zorder=3),
    WayType.GUARD_RAIL: dict(color="#919595", linewidth=1.5, zorder=3),
    WayType.ROAD_BORDER: dict(color="#919595", linewidth=1.5, zorder=3),
    WayType.LINE_THIN_SOLID: dict(color="#FFFFFF", linewidth=0.75, zorder=3),
    WayType.LINE_THIN_SOLID_SOLID: dict(
        color="#FFFFFF", linewidth=0.75, zorder=3
    ),
    WayType.LINE_THIN_DASHED: dict(
        color="#FFFFFF", linewidth=0.75, dashes=[5, 5], zorder=3
    ),
    WayType.LINE_THICK_SOLID: dict(color="#FFFFFF", linewidth=1.5, zorder=3),
    WayType.LINE_THICK_SOLID_SOLID: dict(
        color="#FFFFFF", linewidth=1.5, zorder=3
    ),
    WayType.STOP_LINE: dict(color="#FFFFFF", linewidth=1.5, zorder=3),
    WayType.PEDESTRIAN_MARKING: dict(color="#FFFFFF", linewidth=1.5, zorder=4),
    WayType.VIRTUAL: dict(
        color="#FFFFFF", linewidth=0.25, dashes=[2, 10], zorder=3
    ),
    WayType.VIRTUAL_SOLID: dict(color="#FFFFFF", linewidth=0.25, zorder=3),
    WayType.TRAFFIC_SIGN: dict(color="#D7263D", linewidth=1.5, zorder=4),
}

# Type Aliases
MapLayer = gpd.GeoDataFrame
PathLike = Union[str, "os.PathLike[str]"]


class INTERACTIONMapLayers(Enum):
    """An enumeration of the INTERACTION map layers."""

    NODE = 0
    """The layer of all the map node elements."""
    WAY = 1
    """The layer of all the map way elements."""
    LANELET = 2
    """The layer of all the map lanelet elements."""
    MULTIPOLYGON = 3
    """The layer of all the map multipolygon elements."""
    REGULATORY_ELEMENT = 4
    """The layer of all the map regulatory elements."""

    @classmethod
    def deserialize(
        cls, layer_name: Union[str, "INTERACTIONMapLayers"]
    ) -> "INTERACTIONMapLayers":
        """Deserialize a map layer member from string layer name.

        Args:
            layer_name (str): the string name of the layer.

        Returns:
            INTERACTIONMapLayers: the corresponding map layer member.

        Raises:
            TypeError: if the `layer_name` is not a valid type.
        """
        if isinstance(layer_name, INTERACTIONMapLayers):
            return layer_name
        elif isinstance(layer_name, str):
            return INTERACTIONMapLayers.__members__[layer_name.upper()]
        else:
            raise TypeError(
                f"Invalid `layer_name` to deserialize from: {layer_name}"
            )


class INTERACTIONMap:
    """Map API class for INTERACTION dataset.

    Attributes:
        bounds (Tuple[float, float, float, float]): the map bounding region as
        a Tuple of :obj:`(min_x, min_y, max_x, max_y)`.
        location (str): the string name of the map.
        map_root (str): the root directory of the map data files.
    """

    __slots__ = "_layers", "_root", "_location", "_object_getters"

    def __init__(self, root: PathLike, location: str) -> None:
        """Construct an INTERACTION dataset map API class.

        Args:
            root (PathLike): the root directory of the map data files.
            location (str): the string name of the map.

        Raises:
            AssertionError: if the map data files are invalid.
        """
        self._layers: dict[str, MapLayer] = defaultdict(MapLayer)
        self._root = Path(root).resolve()
        self._location = location
        self._init_layers()

        # sanity checks
        assert (
            len(self._layers[INTERACTIONMapLayers.NODE]) > 0
        ), AssertionError("Empty map node layer.")
        assert len(self._layers[INTERACTIONMapLayers.WAY]) > 0, AssertionError(
            "Empty map way layer."
        )
        assert (
            len(self._layers[INTERACTIONMapLayers.LANELET]) > 0
        ), AssertionError("Empty map lanelet layer.")

        self._object_getters: dict[
            INTERACTIONMapLayers,
            Callable[
                [
                    str,
                ],
                MapElement,
            ],
        ] = {
            INTERACTIONMapLayers.NODE: self._get_node,
            INTERACTIONMapLayers.WAY: self._get_way,
            INTERACTIONMapLayers.LANELET: self._get_lanelet,
            INTERACTIONMapLayers.MULTIPOLYGON: self._get_multipolygon,
            INTERACTIONMapLayers.REGULATORY_ELEMENT: self._get_regulatory_element,
        }

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Tuple[float, float, float, float]: bounding as
        :obj:`(minx, miny, maxx, maxy)`."""
        _nodes = self._layers[INTERACTIONMapLayers.NODE].geometry
        assert isinstance(_nodes, gpd.GeoSeries)
        _union_nodes: MultiPoint = _nodes.unary_union
        assert isinstance(_union_nodes, MultiPoint)
        return _union_nodes.bounds

    @property
    def location(self) -> str:
        """str: the string name of the map location."""
        return self._location

    @property
    def map_root(self) -> str:
        """str: the root directory of map data files."""
        return str(self._root)

    def get_available_layers(self) -> dict[INTERACTIONMapLayers, int]:
        """Returns a Dict mapping each map layer to its number of objects.

        Returns:
            Dict[INTERACTIONMapLayers, int]: a mapping from layer to the number
            of objects in the layer.
        """
        return {
            layer: len(self._layers[layer]) for layer in INTERACTIONMapLayers
        }

    def get_map_layer(self, layer_name: str) -> MapLayer:
        """Returns the map layer.

        Args:
            layer_name (str): name of the map layer.

        Returns:
            MapLayer: the requested map layer.
        """
        return self._layers[INTERACTIONMapLayers.deserialize(layer_name)]

    def get_map_object(
        self,
        object_id: Union[int, str],
        layer: Union[str, INTERACTIONMapLayers],
    ) -> MapElement:
        """Returns a map object of given id in a given layer.

        Args:
            object_id (Union[int, str]): id of the requested map object.
            layer (INTERACTIONMapLayers): map layer the object belongs to.

        Returns:
            MapElement: the requested map object.

        Raises:
            KeyError: if the object ID is invalid.
        """
        layer = INTERACTIONMapLayers.deserialize(layer)
        if int(object_id) not in self._layers[layer].index:
            raise KeyError(f"Invalid object ID: {object_id}.")
        return self._object_getters[layer](int(object_id))

    def get_all_proximal_layers(
        self, loc: Iterable[float], radius: Optional[float] = None
    ) -> Generator[tuple[INTERACTIONMapLayers, MapLayer], None, None]:
        """Returns all the map layers within a query range.

        Args:
            loc (Iterable[float]): the centroid of the query range.
            radius (Optional[float], optional): the radius of the query range
            in meters, and if it is `None`, return all the layers available.
            Defaults to `None`.

        Returns:
            Generator[Tuple[INTERACTIONMapLayers, MapLayer], None, None]:
            two-Tuples with map layer and the proximal map layer within
            query range.
        """
        return self.get_proximal_map_layers(
            loc=loc,
            layers=self.get_available_layers().keys(),
            radius=radius,
        )

    def get_all_proximal_map_objects(
        self, loc: Iterable[float], radius: Optional[float] = None
    ) -> Generator[tuple[INTERACTIONMapLayers, list[MapElement]], None, None]:
        """Returns all the map objects within a query range.

        Args:
            loc (Iterable[float]): the centroid of the query range.
            radius (Optional[float], optional): the radius of the query range
            in meters, and if it is `None`, return all the objects available.
            Defaults to `None`.

        Returns:
            Generator[Tuple[INTERACTIONMapLayers, List[MapElement]], None,
            None]:two-Tuples with map layer and a list of the proximal map
            objects within the map layer that are within query range.
        """
        return self.get_proximal_map_objects_of_layers(
            loc=loc,
            layers=list(INTERACTIONMapLayers.__members__.values()),
            radius=radius,
        )

    def get_proximal_map_layers(
        self,
        loc: Iterable[float],
        layers: list[Union[str, INTERACTIONMapLayers]],
        radius: Optional[float] = None,
    ) -> Generator[tuple[INTERACTIONMapLayers, MapLayer], None, None]:
        """Yields the map layer dataframe clipped into the query range.

        Args:
            loc (Iterable[float]): the centroid of the query range.
            layers (List[Union[str, INTERACTIONMapLayers]]): the list of map
            layers to be queried.
            radius (Optional[float], optional): the radius of the query range
            in meters, and if it is `None`, return all the objects available.
            Defaults to `None`.

        Yields:
            Tuple[INTERACTIONMapLayers, MapLayer]: two-tuples with map layer
            and the corresponding dataframe clipped within the query range.
        """

        for layer in layers:
            if isinstance(layer, str):
                layer = INTERACTIONMapLayers.deserialize(layer)
            assert isinstance(
                layer, INTERACTIONMapLayers
            ), "Invalid map layer."
            df = self.get_map_layer(layer.name)
            if len(df) > 0:
                if radius is not None:
                    buffer = Point(*loc[0:2]).buffer(distance=radius)
                    yield (layer, df[df.geometry.intersects(buffer)])
                else:
                    yield (layer, df)

    def get_proximal_map_objects_of_layers(
        self,
        loc: Iterable[float],
        layers: list[Union[INTERACTIONMapLayers, str]],
        radius: Optional[float] = None,
    ) -> Generator[tuple[INTERACTIONMapLayers, list[MapElement]], None, None]:
        """Yields the map objects within a query range in given map layers.

        Args:
            loc (Iterable[float]): the centroid of the query range.
            layers (List[Union[INTERACTIONMapLayers, str]]): the list of layers
            or the names of layers to query from.
            radius (Optional[float], optional): the radius of the query range
            in meters, and if it is `None`, return all the objects available.

        Yields:
            Tuple[INTERACTIONMapLayers, List[MapElement]]: two-tuples with map
            layer and a list of the proximal map objects in the map layer that
            are within query range.
        """

        for layer in layers:
            if isinstance(layer, str):
                layer = INTERACTIONMapLayers.deserialize(layer)
            assert isinstance(
                layer, INTERACTIONMapLayers
            ), "Invalid map layer."
            df = self.get_map_layer(layer.name)
            if len(df) > 0:
                if radius is not None:
                    buffer = Point(*loc[0:2]).buffer(distance=radius)
                    df = df[df.geometry.intersects(buffer)]
                yield (
                    layer,
                    [
                        self.get_map_object(object_id, layer)
                        for object_id in df.index
                    ],
                )

    def render(
        self,
        anchor: Optional[tuple[float, float, float]] = None,
        radius: Optional[float] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Render the base map.

        Args:
            anchor (Optional[Tuple[float, float, float]], optional): the
            anchor point of the map, and if it is `None`, the map will be
            rendered at the center of the map. Defaults to `None`.
            radius (Optional[float], optional): the radius of the map to be
            rendered in meters, and if it is `None`, the map will be rendered
            with the default size. Defaults to `None`.
            ax (Optional[plt.Axes], optional): the matplotlib axes to render
            the map on, and if it is `None`, a new axes will be created.

        Returns:
            plt.Axes: the matplotlib axes that the map is rendered on.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, dpi=600)

        if anchor is not None:
            assert (
                len(anchor) == 3
            ), "Invalid anchor point: must be a 2D pose with [x, y, heading]."
            x, y, heading = anchor
            cosine, sine = math.cos(heading), math.sin(heading)
            xoff, yoff = cosine * x + sine * y, -sine * x + cosine * y
            affine_params = [cosine, sine, -sine, cosine, -xoff, -yoff]
        else:
            affine_params = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        for layer_type, layer_df in self.get_all_proximal_layers(
            loc=anchor, radius=radius
        ):
            geometry: gpd.GeoDataFrame = layer_df.geometry.copy()
            geometry = geometry.affine_transform(matrix=affine_params)
            if len(geometry) == 0:
                # skip empty layers
                continue

            if layer_type == INTERACTIONMapLayers.NODE:
                geometry.plot(
                    ax=ax, color="k", marker="d", markersize=1, zorder=4
                )
            elif layer_type == INTERACTIONMapLayers.WAY:
                for i, way in layer_df.iterrows():
                    style_dict = WAY_STYLE_MAPPING[WayType[way["type"]]]
                    line = geometry.loc[i]
                    assert isinstance(line, LineString), "Invalid geometry!"
                    ax.plot(line.xy[0], line.xy[1], **style_dict)
            elif layer_type == INTERACTIONMapLayers.LANELET:
                geometry.plot(
                    ax=ax, ec="#FFFFFF", fc="#1D1B18", lw=0.0, zorder=2
                )
            elif layer_type == INTERACTIONMapLayers.MULTIPOLYGON:
                geometry.plot(
                    ax=ax,
                    ec="#FFFFFF",
                    fc="#737373",
                    lw=0.0,
                    alpha=0.75,
                    zorder=2,
                )
            else:
                continue

        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.get_figure().tight_layout()
        ax.get_figure().set_dpi(600)
        ax.get_figure().set_facecolor("#000000")

        return ax

    def _init_layers(self) -> None:
        map_file = Path(self._root).joinpath(Path(self._location + ".osm"))
        map_tree = safe_parse(map_file)
        _layer_initializers = {
            INTERACTIONMapLayers.NODE: self._init_node_layer,
            INTERACTIONMapLayers.WAY: self._init_way_layer,
            INTERACTIONMapLayers.LANELET: self._init_lanelet_layer,
            INTERACTIONMapLayers.MULTIPOLYGON: self._init_multipolygon_layer,
            INTERACTIONMapLayers.REGULATORY_ELEMENT: self._init_regulatory_layer,
        }
        for layer, initializer in _layer_initializers.items():
            self._layers[layer] = initializer(map_tree)
        self._post_update_stop_lines()

    def _init_node_layer(
        self, map_tree: etree.ElementTree.ElementTree
    ) -> MapLayer:
        node_list = []
        for node in map_tree.findall("node"):
            projector = INTERACTIONProjector()
            # ignore non-visible nodes
            if node.get("visible") != "true":
                continue

            object_id = int(node.get("id"))
            lng, lat = float(node.get("lon")), float(node.get("lat"))
            x, y = projector(lng, lat)

            node_list.append({"object_id": object_id, "geometry": Point(x, y)})

        df = gpd.GeoDataFrame(node_list)
        if "object_id" in df.columns:
            df.set_index("object_id", inplace=True)

        return df

    def _init_way_layer(
        self, map_tree: etree.ElementTree.ElementTree
    ) -> MapLayer:
        way_list = []
        for way in map_tree.findall("way"):
            # ignore non-visible way elements
            if way.get("visible") != "true":
                continue

            object_id = int(way.get("id"))
            type_str = way.find("tag[@k='type']").get("v")
            subtype_str = way.find("tag[@k='subtype']")
            if isinstance(subtype_str, etree.ElementTree.Element):
                subtype_str = subtype_str.get("v")
            else:
                subtype_str = None
            way_type = get_way_type(type_str, subtype_str)

            node_ids = [nd.get("ref") for nd in way.findall("nd")]
            way = instantiate_way(map_tree, object_id)

            way_list.append(
                {
                    "object_id": object_id,
                    "type": way_type.name,
                    "node_ids": node_ids,
                    "geometry": way,
                }
            )

        df = gpd.GeoDataFrame(way_list)
        if "object_id" in df.columns:
            df.set_index("object_id", inplace=True)

        return df

    def _init_lanelet_layer(
        self, map_tree: etree.ElementTree.ElementTree
    ) -> MapLayer:
        lanelet_list = []
        for lanelet in map_tree.findall("relation/tag[@v='lanelet']/.."):
            # ignore non-visible lanelet
            if not lanelet.get("visible") == "true":
                continue
            object_id = int(lanelet.get("id"))

            # parse lanelet subtype
            subtype = lanelet.find("tag[@k='subtype']")
            if isinstance(subtype, etree.ElementTree.Element):
                subtype = LaneletSubType[subtype.get("v").upper()]
            else:
                subtype = LaneletSubType.UNDEFINED

            # parase boundaries
            left_boundary_id = int(
                lanelet.find("member[@role='left']").get("ref")
            )
            left_boundary = self._get_way(left_boundary_id)
            right_boundary_id = int(
                lanelet.find("member[@role='right']").get("ref")
            )
            right_boundary = self._get_way(right_boundary_id)

            # pre-transformations
            if object_id in LANELET_TO_REVERSE[self._location]:
                right_boundary = reverse_way(right_boundary)

            left_boundary_dir = get_linestring_direction(
                left_boundary.to_geometry()
            )
            right_boundary_dir = get_linestring_direction(
                right_boundary.to_geometry()
            )
            if (
                left_boundary_dir[0] * right_boundary_dir[0]
                + left_boundary_dir[1] * right_boundary_dir[1]
                <= 0
            ):
                left_boundary = reverse_way(left_boundary)

            # parse speed limit
            speed_limit = None
            for reg in lanelet.findall("member[@role='regulatory_element']"):
                reg = map_tree.find(f"relation[@id='{reg.get('ref')}']")
                if reg.find("tag[@k='subtype']").get("v") == "speed_limit":
                    assert speed_limit is None, RuntimeError(
                        f"Failed to parse lanelet {object_id}: "
                        "Duplicated speed limits!"
                    )
                    # get speed limit
                    speed_limit = reg.find("tag[@k='sign_type']").get("v")
                    speed_limit = SpeedLimit.from_string(speed_limit)
            assert speed_limit is not None, RuntimeError(
                f"Failed to parse lanelet {object_id}: Missing speed limit!"
            )

            lanelet_list.append(
                {
                    "object_id": object_id,
                    "subtype": subtype.name,
                    "speed_limit": speed_limit.speed_limit_mps,
                    "left_boundary_id": int(left_boundary_id),
                    "left_boundary": left_boundary,
                    "right_boundary_id": int(right_boundary_id),
                    "right_boundary": right_boundary,
                    "stop_line_id": None,
                    "stop_line": None,
                    "adjacent_lanelets": None,
                    "preceding_lanelets": None,
                    "succeeding_lanelets": None,
                    "geometry": Polygon(
                        [
                            coord
                            for coord in chain(
                                substring(
                                    left_boundary.to_geometry(), 1, 0, True
                                ).coords,
                                right_boundary.to_geometry().coords,
                            )
                        ]
                    ),
                }
            )

        df = gpd.GeoDataFrame(lanelet_list)
        if "object_id" not in df.columns:
            return df
        df.set_index("object_id", inplace=True)

        # create lanelet connectivity
        def _get_heads_and_tails(row: gpd.GeoSeries) -> tuple[Point, ...]:
            left_head = Point(row["left_boundary"].to_geometry().coords[0])
            left_tail = Point(row["left_boundary"].to_geometry().coords[-1])
            right_head = Point(row["right_boundary"].to_geometry().coords[0])
            right_tail = Point(row["right_boundary"].to_geometry().coords[-1])

            return left_head, left_tail, right_head, right_tail

        for i, q_row in df.iterrows():
            adjacent_lanelet_ids = []
            preceding_lanelet_ids = []
            succeeding_lanelet_ids = []

            for j, v_row in df.iterrows():
                if (
                    q_row["left_boundary_id"] == v_row["right_boundary_id"]
                    or q_row["right_boundary_id"] == v_row["left_boundary_id"]
                ):
                    adjacent_lanelet_ids.append(j)
                    continue
                qht = _get_heads_and_tails(q_row)
                vht = _get_heads_and_tails(v_row)
                if math.isclose(
                    qht[1].distance(vht[0]), 0.0, abs_tol=1e-3
                ) and math.isclose(qht[3].distance(vht[2]), 0.0, abs_tol=1e-3):
                    succeeding_lanelet_ids.append(j)
                    continue
                if math.isclose(
                    qht[0].distance(vht[1]), 0.0, abs_tol=1e-3
                ) and math.isclose(qht[2].distance(vht[3]), 0.0, abs_tol=1e-3):
                    preceding_lanelet_ids.append(j)
                    continue

            df.at[i, "adjacent_lanelets"] = adjacent_lanelet_ids
            df.at[i, "preceding_lanelets"] = preceding_lanelet_ids
            df.at[i, "succeeding_lanelets"] = succeeding_lanelet_ids

        return df

    def _init_multipolygon_layer(
        self, map_tree: etree.ElementTree.ElementTree
    ) -> MapLayer:
        mp_list = []
        for mp in map_tree.findall("relation/tag[@v='multipolygon']/.."):
            # ignore non-visible multipolygon elements
            if mp.get("visible") != "true":
                continue
            object_id = int(mp.get("id"))

            subtype = mp.find("tag[@k='subtype']")
            if isinstance(subtype, etree.ElementTree.Element):
                subtype = MultiPolygonSubType[subtype.get("v").upper()]
            else:
                subtype = MultiPolygonSubType.UNDEFINED

            outer_way_ids: list[str] = []
            outer_ways: list[Way] = []
            for way in mp.findall("member[@role='outer']"):
                outer_way_ids.append(way.get("ref"))
                outer_ways.append(self._get_way(int(way.get("ref"))))
            outer_ways = order_ways(outer_ways)

            mp_list.append(
                {
                    "object_id": object_id,
                    "subtype": subtype.name,
                    "outer_id": outer_way_ids,
                    "outer_way": MultiLineString(
                        [way.to_geometry() for way in outer_ways]
                    ),
                    "geometry": Polygon(
                        [
                            coord
                            for way in outer_ways
                            for coord in way.to_geometry().coords
                        ]
                    ),
                }
            )

        df = gpd.GeoDataFrame(mp_list)
        if "object_id" in df.columns:
            df.set_index("object_id", inplace=True)

        return df

    def _init_regulatory_layer(
        self, map_tree: etree.ElementTree.ElementTree
    ) -> MapLayer:
        reg_list = []
        for reg in map_tree.findall(
            "relation/tag[@v='regulatory_element']/.."
        ):
            # ignore non-visible regulatory elements and speed limits
            if (
                reg.get("visible") != "true"
                or reg.find("tag[@v='speed_limit']") is not None
            ):
                continue

            object_id = int(reg.get("id"))
            # parse subtype
            subtype = reg.find("tag[@k='subtype']")
            if isinstance(subtype, etree.ElementTree.Element):
                subtype = RegulatoryElementSubType[subtype.get("v").upper()]
            else:
                subtype = RegulatoryElementSubType.UNDEFINED

            prior_lane_ids = []
            yield_lane_ids = []
            for prior_lane in reg.findall("member[@role='right_of_way']"):
                prior_lane_ids.append(int(prior_lane.get("ref")))
            for yield_lane in reg.findall("member[@role='yield']"):
                yield_lane_ids.append(int(yield_lane.get("ref")))

            refer_ids: list[int] = []
            refers: list[LineString] = []
            ref_line_ids: list[int] = []
            for refer in reg.findall("member[@role='refers']"):
                # validate refer line type
                refer_id = int(refer.get("ref"))
                refer = map_tree.find(f"way[@id='{refer_id}']")
                if refer.find("tag[@k='type']").get("v") != "traffic_sign":
                    # handle ref_line mislabeling
                    ref_line_ids.append(refer_id)
                else:
                    refer_ids.append(refer_id)
                    refers.append(instantiate_way(map_tree, refer_id))

            for ref_line in reg.findall("member[@role='ref_line']"):
                # validate ref_line type
                ref_line_id = int(ref_line.get("ref"))
                ref_line = map_tree.find(f"way[@id='{ref_line_id}']")
                if ref_line.find("tag[@k='type']").get("v") == "traffic_sign":
                    refer_ids.append(ref_line_id)
                    refers.append(instantiate_way(map_tree, ref_line_id))
                else:
                    ref_line_ids.append(ref_line_id)

            reg_list.append(
                {
                    "object_id": object_id,
                    "subtype": subtype.name,
                    "prior_lane_ids": prior_lane_ids,
                    "yield_lane_ids": yield_lane_ids,
                    "refer_line_ids": refer_ids,
                    "ref_line_ids": ref_line_ids,
                    "geometry": Polygon(
                        [coord for line in refers for coord in line.coords]
                    ),
                }
            )

        df = gpd.GeoDataFrame(reg_list)
        if "object_id" in df.columns:
            df.set_index("object_id", inplace=True)

        return df

    def _post_update_stop_lines(self) -> None:
        df = self._layers[INTERACTIONMapLayers.LANELET].copy()
        for _id, row in self._layers[
            INTERACTIONMapLayers.REGULATORY_ELEMENT
        ].iterrows():
            ref_line_ids = row["ref_line_ids"]
            yield_lane_ids = row["yield_lane_ids"]
            assert (
                len(ref_line_ids) == len(yield_lane_ids)
                or len(ref_line_ids) == 1
            ), ValueError(
                f"Error post-processing regulatory element {_id}: "
                "Expect the number of 'ref_line' to be no more than "
                "the number of 'yield_lane', "
                f"but got {len(ref_line_ids)} and {len(yield_lane_ids)}."
            )
            if len(ref_line_ids) == 1:
                r_id = int(ref_line_ids[0])
                for l_id in yield_lane_ids:
                    l_id = int(l_id)
                    df.loc[l_id, "stop_line_id"] = r_id
                    df.loc[l_id, "stop_line"] = self._get_way(r_id)
            else:
                for r_id, l_id in zip(ref_line_ids, yield_lane_ids):
                    l_id, r_id = int(l_id), int(r_id)
                    df.loc[l_id, "stop_line_id"] = r_id
                    df.loc[l_id, "stop_line"] = self._get_way(r_id)
        self._layers[INTERACTIONMapLayers.LANELET] = df

    def _get_node(self, object_id: Union[int, str]) -> Node:
        df = self._layers[INTERACTIONMapLayers.NODE]
        point = df.loc[int(object_id)].geometry
        assert isinstance(point, Point)
        return Node(id=int(object_id), x=point.x, y=point.y)

    def _get_way(self, object_id: Union[int, str]) -> Way:
        df = self._layers[INTERACTIONMapLayers.WAY]
        df = df.loc[int(object_id)]
        return Way(
            id=int(object_id),
            type=WayType[df["type"].upper()],
            nodes=[self._get_node(node_id) for node_id in df["node_ids"]],
        )

    def _get_lanelet(self, object_id: Union[int, str]) -> Lanelet:
        df = self._layers[INTERACTIONMapLayers.LANELET]
        df = df.loc[int(object_id)]
        return Lanelet(
            id=int(object_id),
            subtype=LaneletSubType[df["subtype"]],
            speed_limit=SpeedLimit.from_mps(df["speed_limit"]),
            left=df["left_boundary"],
            right=df["right_boundary"],
            stop_line=df["stop_line"],
            adjacent_lanelets=df["adjacent_lanelets"],
            preceding_lanelets=df["preceding_lanelets"],
            succeeding_lanelets=df["succeeding_lanelets"],
        )

    def _get_multipolygon(
        self, object_id: Union[int, str]
    ) -> Optional[MultiPolygon]:
        df = self._layers[INTERACTIONMapLayers.MULTIPOLYGON]
        df = df.loc[int(object_id)]
        if len(df) == 0:
            return None

        return MultiPolygon(
            id=int(object_id),
            subtype=MultiPolygonSubType[df["subtype"].upper()],
            outer=[self._get_way(outer_id) for outer_id in df["outer_id"]],
        )

    def _get_regulatory_element(self, object_id: str) -> RegulatoryElement:
        df = self._layers[INTERACTIONMapLayers.REGULATORY_ELEMENT]
        df = df.loc[int(object_id)]
        return RegulatoryElement(
            id=int(object_id),
            subtype=RegulatoryElementSubType[df["subtype"].upper()],
            refers=[
                self._get_way(int(way_id)) for way_id in df["refer_line_ids"]
            ],
            prior_lanelets=df["prior_lane_ids"],
            yield_lanelets=df["yield_lane_ids"],
        )

    def __str__(self) -> str:
        return (
            f"<INTERACTIONMap(location={self._location}, "
            f"root={self._root}) at {hex(id(self))}>"
        )

    def __repr__(self) -> str:
        return str(self)
