"""Utilities for the maps dataset."""
# Copyright (c) 2023, Juanwu Lu <juanwu@purdue.edu>.
# Released under the BSD-3-Clause license.
# See https://opensource.org/license/bsd-3-clause/ for licensing details.
import math
from collections.abc import Iterable
from typing import Optional, Union
from xml import etree

from shapely import LineString, Point

from .elements import Way
from .projector import INTERACTIONProjector
from .typing import WayType

# NOTE: the list of lanlet elements to revers only applies to version 1.1
LANELET_TO_REVERSE: dict[str, list[int]] = {
    "DR_CHN_Merging_ZS0": [
        30014,
        30025,
        30026,
        30030,
        30033,
        30034,
        30038,
        30041,
        30044,
        30047,
    ],
    "DR_CHN_Merging_ZS2": [
        30014,
        30025,
        30026,
        30030,
        30033,
        30034,
        30038,
        30041,
        30044,
        30047,
    ],
    "DR_CHN_Roundabout_LN": [
        30001,
        30002,
        30004,
        30007,
        30008,
        30011,
        30012,
        30016,
        30020,
        30026,
        30029,
        30032,
        30035,
        30037,
        30041,
        30046,
        30048,
        30051,
        30058,
        30059,
        30060,
        30061,
        30062,
        30063,
        30064,
        30065,
        30066,
        30068,
        30069,
        30070,
        30071,
        30072,
        30073,
        30077,
        30081,
        30083,
        30084,
        30087,
        30089,
        30090,
        30091,
        30092,
        30093,
    ],
    "DR_DEU_Merging_MT": [
        30001,
        30002,
        30004,
        30005,
        30006,
        30007,
        30008,
        30010,
        30011,
        30012,
    ],
    "DR_DEU_Roundabout_OF": [
        30000,
        30001,
        30002,
        30003,
        30004,
        30005,
        30006,
        30009,
        30011,
        30013,
        30015,
        30016,
        30017,
        30018,
        30019,
        30020,
        30023,
        30025,
        30026,
        30027,
        30028,
        30030,
        30031,
        30033,
        30034,
        30035,
        30036,
        30037,
        30039,
        30041,
        30043,
        30044,
    ],
    "DR_Intersection_CM": [
        30000,
        30008,
        30009,
        30010,
        30017,
        30018,
        30019,
        30020,
        30022,
        30023,
        30025,
        30026,
        30027,
    ],
    "DR_LaneChange_ET0": [
        30001,
        30003,
        30004,
        30005,
        30006,
        30020,
        30021,
        30022,
        30023,
        30026,
        30027,
        30028,
        30029,
        30031,
        30032,
        30033,
        30034,
        30036,
        30040,
        30059,
        30060,
        30071,
        30073,
        30076,
        30077,
        30078,
        30079,
    ],
    "DR_LaneChange_ET1": [
        30001,
        30003,
        30004,
        30005,
        30006,
        30020,
        30021,
        30022,
        30023,
        30026,
        30027,
        30028,
        30029,
        30031,
        30032,
        30033,
        30034,
        30036,
        30040,
        30059,
        30060,
        30071,
        30073,
        30076,
        30077,
        30078,
        30079,
    ],
    "DR_Merging_TR0": [
        30000,
        30002,
        30004,
        30007,
        30009,
        30011,
        30012,
        30016,
        30017,
        30018,
        30019,
        30024,
        30025,
        30027,
        30029,
        30034,
        30037,
        30039,
        30040,
        30041,
        30051,
        30052,
        30054,
        30055,
    ],
    "DR_Merging_TR1": [
        30000,
        30002,
        30004,
        30007,
        30009,
        30011,
        30012,
        30016,
        30017,
        30018,
        30019,
        30024,
        30025,
        30027,
        30029,
        30034,
        30037,
        30039,
        30040,
        30041,
        30051,
        30052,
        30054,
        30055,
    ],
    "DR_Roundabout_RW": [30001, 30009, 30011, 30013, 30028],
    "DR_USA_Intersection_EP0": [
        30002,
        30004,
        30006,
        30010,
        30011,
        30015,
        30016,
        30021,
        30024,
        30025,
        30027,
        30028,
        30035,
        30036,
        30038,
        30039,
        30040,
        30049,
        30050,
        30055,
        30056,
        30058,
    ],
    "DR_USA_Intersection_EP1": [
        30002,
        30004,
        30006,
        30010,
        30011,
        30015,
        30016,
        30021,
        30024,
        30025,
        30027,
        30028,
        30035,
        30036,
        30038,
        30039,
        30040,
        30049,
        30050,
        30055,
        30056,
        30058,
    ],
    "DR_USA_Intersection_GL": [],
    "DR_USA_Intersection_MA": [30023, 30027, 30058],
    "DR_USA_Roundabout_EP": [
        30001,
        30002,
        30006,
        30008,
        30009,
        30012,
        30014,
        30015,
        30017,
        30020,
        30027,
        30030,
        30036,
        30043,
        30045,
        30047,
        30048,
        30050,
        30051,
        30052,
        30056,
        30057,
    ],
    "DR_USA_Roundabout_FT": [
        30003,
        30004,
        30013,
        30018,
        30020,
        30022,
        30028,
        30039,
        30040,
        30047,
    ],
    "DR_USA_Roundabout_SR": [
        30006,
        30008,
        30010,
        30016,
        30018,
        30019,
        30021,
        30026,
        30027,
        30028,
        30029,
        30030,
        30032,
        30034,
        30040,
        30045,
    ],
}


def get_linestring_direction(line: LineString) -> tuple[float, float]:
    """Computes the representative direction vector of a line :obj:`[dx, dy]`.

    Args:
        line (LineString): the geometric linestring to extract direction.

    Returns:
        Tuple[float, float]: a two-tuple direction vector `[dx, dy]`.
    """
    assert isinstance(line, LineString), TypeError(
        f"Expect input `line` to be a `LineString`, but got {type(line)}."
    )
    representative_begin = line.line_interpolate_point(0.5, True)
    assert isinstance(representative_begin, Point)
    representative_end = line.line_interpolate_point(0.501, True)
    assert isinstance(representative_end, Point)

    return (
        representative_end.x - representative_begin.x,
        representative_end.y - representative_begin.y,
    )


def get_way_type(type_str: str, subtype_str: str) -> WayType:
    """Returns the `WayType` enum member given type and subtype strings.

    Args:
        type_str (str): the type string of the current `Way` element.
        subtype_str (str): the subtype string of the current `Way` element.

    Returns:
        WayType: the `WayType` enum member of the current `Way` element.
    """
    if type_str == "traffic_sign" or subtype_str is None:
        way_type_str = type_str.upper()
    else:
        way_type_str = "_".join([type_str.upper(), subtype_str.upper()])

    return WayType[way_type_str]


def instantiate_way(
    map_tree: etree.ElementTree.ElementTree, element_id: Union[int, str]
) -> Optional[LineString]:
    """Instantiates a `LineString` object from a `way` element in raw map tree.

    Args:
        map_tree (xml.etree.ElementTree.ElementTree): the raw map tree.
        element_id (Union[int, str]): the id of the `way` element.

    Returns:
        Optional[LineString]: the `LineString` object instantiated from the
        `way` element, or `None` if the `way` element is not visible.
    """
    projector = INTERACTIONProjector()
    assert isinstance(map_tree, etree.ElementTree.ElementTree)
    way_element = map_tree.find(f"way[@id='{element_id}']")
    if way_element.get("visible") != "true":
        # ignore non-visible way element
        return None

    coordinates: list[tuple[float, float]] = []
    for nd_ref in way_element.findall("nd"):
        node = map_tree.find(f"node[@id='{nd_ref.get('ref')}']")
        if node.get("visible") != "true":
            # ignore non-visible node elements
            continue
        lng, lat = float(node.get("lon")), float(node.get("lat"))
        if lng is None or lat is None:
            # handle `.osm_xy` file format
            x, y = float(node.get("x")), float(node.get("y"))
        else:
            x, y = projector(lng, lat)
        assert x is not None and y is not None
        coordinates.append((x, y))

    return LineString(coordinates=coordinates)


def order_ways(ways: Iterable[Way]) -> list[Way]:
    """Orders an iterable of `Way` elements to form a continuous path.

    Args:
        ways (Iterable[Way]): the iterable of `Way` elements to order.

    Returns:
        List[Way]: the ordered list of `Way` elements.
    """
    if len(ways) == 0:
        return []

    # start with the first way element
    ordered_ways: list[Way] = [ways.pop(0)]

    # while there are still way element left
    while ways:
        # Find the element whose start is closest to the last linestring's end
        last_end = ordered_ways[-1].to_geometry().coords[-1]
        min_distance = math.inf
        next_index = None

        # Iterate through the remaining way elements
        for i, way in enumerate(ways):
            distance = math.hypot(
                way.to_geometry().coords[0][0] - last_end[0],
                way.to_geometry().coords[0][1] - last_end[1],
            )
            if distance < min_distance:
                min_distance = distance
                next_index = i

        ordered_ways.append(ways.pop(next_index))

    return ordered_ways


def reverse_way(way: Way) -> Way:
    """Reverses the direction and returns a new `Way` element.

    Args:
        way (Way): the `Way` element to reverse.

    Returns:
        Way: the reversed `Way` element.
    """
    assert isinstance(way, Way), TypeError(
        f"Expect inptu `way` to be a `Way`, but got {way.__class__.__name__}."
    )
    return Way(
        id=way.id,
        type=way.type,
        nodes=[node for node in way.nodes[::-1]],
    )
