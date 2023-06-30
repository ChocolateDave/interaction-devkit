import pytest
from shapely.geometry import LineString, Point, Polygon

from interaction.dataset.maps.elements import (
    Lanelet,
    MultiPolygon,
    Node,
    RegulatoryElement,
    SpeedLimit,
    Way,
)
from interaction.dataset.maps.projector import INTERACTIONProjector
from interaction.dataset.maps.typing import (
    LaneletSubType,
    MultiPolygonSubType,
    RegulatoryElementSubType,
    RightOfWay,
    WayType,
)

# Constants for testing
ABS_TOL: float = 1e-6


# ----------- Test cases for element typing ----------- #
def test_way_type() -> None:
    """Test the `WayType` enumeration."""
    assert WayType.UNDEFINED == 0
    assert WayType.CURBSTONE_LOW == 1
    assert WayType.GUARD_RAIL == 2
    assert WayType.ROAD_BORDER == 3
    assert WayType.LINE_THIN_SOLID == 4
    assert WayType.LINE_THIN_SOLID_SOLID == 4
    assert WayType.LINE_THIN_DASHED == 5
    assert WayType.LINE_THICK_SOLID == 6
    assert WayType.LINE_THICK_SOLID_SOLID == 6
    assert WayType.STOP_LINE == 7
    assert WayType.PEDESTRIAN_MARKING == 8
    assert WayType.VIRTUAL == 9
    assert WayType.VIRTUAL_SOLID == 9
    assert WayType.TRAFFIC_SIGN == 10
    assert WayType.UNDEFINED.one_hot_serialize() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert WayType.CURBSTONE_LOW.one_hot_serialize() == [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert WayType.GUARD_RAIL.one_hot_serialize() == [
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert WayType.ROAD_BORDER.one_hot_serialize() == [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert WayType.LINE_THIN_SOLID.one_hot_serialize() == [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert WayType.LINE_THIN_SOLID_SOLID.one_hot_serialize() == [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert WayType.LINE_THIN_DASHED.one_hot_serialize() == [
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
    ]
    assert WayType.LINE_THICK_SOLID.one_hot_serialize() == [
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    ]
    assert WayType.LINE_THICK_SOLID_SOLID.one_hot_serialize() == [
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    ]
    assert WayType.STOP_LINE.one_hot_serialize() == [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
    ]
    assert WayType.PEDESTRIAN_MARKING.one_hot_serialize() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
    ]
    assert WayType.VIRTUAL.one_hot_serialize() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
    ]
    assert WayType.VIRTUAL_SOLID.one_hot_serialize() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
    ]
    assert WayType.TRAFFIC_SIGN.one_hot_serialize() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
    ]


def test_lanelet_subtype() -> None:
    assert LaneletSubType.UNDEFINED == 0
    assert LaneletSubType.ROAD == 1
    assert LaneletSubType.HIGHWAY == 2

    assert LaneletSubType.UNDEFINED.one_hot_serialize() == [0, 0]
    assert LaneletSubType.ROAD.one_hot_serialize() == [1, 0]
    assert LaneletSubType.HIGHWAY.one_hot_serialize() == [0, 1]


def test_multipolygon_subtype() -> None:
    assert MultiPolygonSubType.UNDEFINED == 0
    assert MultiPolygonSubType.KEEPOUT == 1

    assert MultiPolygonSubType.UNDEFINED.one_hot_serialize() == [0]
    assert MultiPolygonSubType.KEEPOUT.one_hot_serialize() == [1]


def test_regulatory_element_subtype() -> None:
    assert RegulatoryElementSubType.UNDEFINED == 0
    assert RegulatoryElementSubType.SPEED_LIMIT == 1
    assert RegulatoryElementSubType.RIGHT_OF_WAY == 2
    assert RegulatoryElementSubType.ALL_WAY_STOP == 3

    assert RegulatoryElementSubType.UNDEFINED.one_hot_serialize() == [0, 0, 0]
    assert RegulatoryElementSubType.SPEED_LIMIT.one_hot_serialize() == [
        1,
        0,
        0,
    ]
    assert RegulatoryElementSubType.RIGHT_OF_WAY.one_hot_serialize() == [
        0,
        1,
        0,
    ]
    assert RegulatoryElementSubType.ALL_WAY_STOP.one_hot_serialize() == [
        0,
        0,
        1,
    ]


def test_right_of_way() -> None:
    assert RightOfWay.UNDEFINED == 0
    assert RightOfWay.YIELD == 1
    assert RightOfWay.RIGHT_OF_WAY == 2

    assert RightOfWay.UNDEFINED.one_hot_serialize() == [0, 0]
    assert RightOfWay.YIELD.one_hot_serialize() == [1, 0]
    assert RightOfWay.RIGHT_OF_WAY.one_hot_serialize() == [0, 1]


# ----------- Test cases for map projector ----------- #
def test_map_projector() -> None:
    tmp_projector = INTERACTIONProjector(0.0, 0.0)
    assert tmp_projector.origin == pytest.approx((166021.44, 0.0), abs=0.01)
    assert tmp_projector.origin_x == pytest.approx(166021.44, abs=0.01)
    assert tmp_projector.origin_y == pytest.approx(0.00, abs=0.01)
    assert all(
        tmp_projector(*inputs) == pytest.approx(target, abs=0.01)
        for inputs, target in zip(
            (
                (0.00917192296, 0.00860598684),
                (0.00902404929, 0.00863912852),
                (0.00894550787, 0.00866281977),
            ),
            (
                (1022.01490, 952.52631),
                (1005.53769, 956.19462),
                (996.78602, 958.81688),
            ),
        )
    ), "Projector is not working as expected."


# ----------- Test cases for element containers ----------- #
def test_speed_limit() -> None:
    with pytest.raises(RuntimeError):
        SpeedLimit(0.0)

    tmp_speed_limit = SpeedLimit.from_mps(10.0)
    assert tmp_speed_limit.speed_limit_mps == 10.0
    assert tmp_speed_limit.speed_limit_kmph == 36.0
    assert tmp_speed_limit.speed_limit_mph == pytest.approx(22.36, abs=0.01)

    tmp_speed_limit_str = "80kmh"
    tmp_speed_limit = SpeedLimit.from_string(tmp_speed_limit_str)
    assert tmp_speed_limit.speed_limit_mps == pytest.approx(22.22, abs=0.01)
    assert tmp_speed_limit.speed_limit_kmph == pytest.approx(80.00, abs=0.01)
    assert tmp_speed_limit.speed_limit_mph == pytest.approx(49.71, abs=0.01)


def test_node() -> None:
    tmp_node = Node(1000, 1022.01490, 952.52631)
    assert tmp_node.id == 1000
    assert tmp_node.x == 1022.01490
    assert tmp_node.y == 952.52631
    assert Node.deserialize(tmp_node.serialize()) == tmp_node
    assert tmp_node.to_geometry() == Point(1022.01490, 952.52631)


def test_way() -> None:
    tmp_way = Way(
        10000,
        WayType.GUARD_RAIL,
        [Node(1098, 1031.78648, 960.62575), Node(1038, 1024.26416, 961.88454)],
    )
    assert tmp_way.id == 10000
    assert tmp_way.type == WayType.GUARD_RAIL
    assert tmp_way.nodes[0].id == 1098
    assert tmp_way.nodes[0].x == 1031.78648
    assert tmp_way.nodes[0].y == 960.62575
    assert tmp_way.nodes[1].id == 1038
    assert tmp_way.nodes[1].x == 1024.26416
    assert tmp_way.nodes[1].y == 961.88454
    assert Way.deserialize(tmp_way.serialize()) == tmp_way
    assert tmp_way.to_geometry() == LineString(
        [(1031.78648, 960.62575), (1024.26416, 961.88454)]
    )


def test_lanelet() -> None:
    tmp_lanlet = Lanelet(
        id=30046,
        subtype=LaneletSubType.ROAD,
        left=Way(
            10055,
            WayType.LINE_THIN_DASHED,
            [
                Node(1457, 1066.29163, 966.74386),
                Node(1658, 1056.41086, 973.33936),
            ],
        ),
        right=Way(
            10144,
            WayType.CURBSTONE_LOW,
            [
                Node(1160, 1069.40678, 970.41154),
                Node(1059, 1059.66565, 976.67956),
            ],
        ),
        speed_limit=SpeedLimit.from_mph(25.0),
    )
    assert tmp_lanlet.id == 30046
    assert tmp_lanlet.subtype == LaneletSubType.ROAD
    assert tmp_lanlet.left.id == 10055
    assert tmp_lanlet.left.type == WayType.LINE_THIN_DASHED
    assert tmp_lanlet.left.nodes[0].id == 1457
    assert tmp_lanlet.left.nodes[0].x == 1066.29163
    assert tmp_lanlet.left.nodes[0].y == 966.74386
    assert tmp_lanlet.left.nodes[1].id == 1658
    assert tmp_lanlet.left.nodes[1].x == 1056.41086
    assert tmp_lanlet.left.nodes[1].y == 973.33936
    assert tmp_lanlet.right.id == 10144
    assert tmp_lanlet.right.type == WayType.CURBSTONE_LOW
    assert tmp_lanlet.right.nodes[0].id == 1160
    assert tmp_lanlet.right.nodes[0].x == 1069.40678
    assert tmp_lanlet.right.nodes[0].y == 970.41154
    assert tmp_lanlet.right.nodes[1].id == 1059
    assert tmp_lanlet.right.nodes[1].x == 1059.66565
    assert tmp_lanlet.right.nodes[1].y == 976.67956
    assert tmp_lanlet.speed_limit.speed_limit_mph == pytest.approx(
        25.0, abs=0.01
    )
    assert tmp_lanlet.speed_limit.speed_limit_kmph == pytest.approx(
        40.23, abs=0.01
    )
    assert tmp_lanlet.speed_limit.speed_limit_mps == pytest.approx(
        11.18, abs=0.01
    )
    assert Lanelet.deserialize(tmp_lanlet.serialize()) == tmp_lanlet
    assert tmp_lanlet.to_geometry() == Polygon(
        [
            (1069.40678, 970.41154),
            (1059.66565, 976.67956),
            (1056.41086, 973.33936),
            (1066.29163, 966.74386),
        ]
    )


def test_multipolygon() -> None:
    tmp_multipolygon = MultiPolygon(
        40005,
        subtype=MultiPolygonSubType.KEEPOUT,
        outer=[
            Way(
                10110,
                WayType.LINE_THIN_SOLID,
                [
                    Node(1208, 1011.47179, 1005.93759),
                    Node(1210, 1010.77092, 1005.17244),
                    Node(1604, 1010.22275, 1004.47913),
                    Node(1292, 1010.15340, 1004.33972),
                ],
            )
        ],
    )
    assert tmp_multipolygon.id == 40005
    assert tmp_multipolygon.subtype == MultiPolygonSubType.KEEPOUT
    assert tmp_multipolygon.outer[0].id == 10110
    assert tmp_multipolygon.outer[0].type == WayType.LINE_THIN_SOLID
    assert tmp_multipolygon.outer[0].nodes[0].id == 1208
    assert tmp_multipolygon.outer[0].nodes[0].x == 1011.47179
    assert tmp_multipolygon.outer[0].nodes[0].y == 1005.93759
    assert tmp_multipolygon.outer[0].nodes[1].id == 1210
    assert tmp_multipolygon.outer[0].nodes[1].x == 1010.77092
    assert tmp_multipolygon.outer[0].nodes[1].y == 1005.17244
    assert tmp_multipolygon.outer[0].nodes[2].id == 1604
    assert tmp_multipolygon.outer[0].nodes[2].x == 1010.22275
    assert tmp_multipolygon.outer[0].nodes[2].y == 1004.47913
    assert tmp_multipolygon.outer[0].nodes[3].id == 1292
    assert tmp_multipolygon.outer[0].nodes[3].x == 1010.15340
    assert tmp_multipolygon.outer[0].nodes[3].y == 1004.33972
    assert (
        MultiPolygon.deserialize(tmp_multipolygon.serialize())
        == tmp_multipolygon
    )
    assert tmp_multipolygon.to_geometry() == Polygon(
        [
            (1011.47179, 1005.93759),
            (1010.77092, 1005.17244),
            (1010.22275, 1004.47913),
            (1010.15340, 1004.33972),
        ]
    )


def test_regulatory_element() -> None:
    tmp_regulatory_element = RegulatoryElement(
        id=50001,
        subtype=RegulatoryElementSubType.RIGHT_OF_WAY,
        refers=[
            Way(
                10060,
                WayType.TRAFFIC_SIGN,
                [
                    Node(1622, 993.28081, 981.69366),
                    Node(1631, 993.34466, 981.02774),
                    Node(1625, 993.40880, 980.35882),
                ],
            )
        ],
    )
    assert tmp_regulatory_element.id == 50001
    assert (
        tmp_regulatory_element.subtype == RegulatoryElementSubType.RIGHT_OF_WAY
    )
    assert tmp_regulatory_element.refers[0].id == 10060
    assert tmp_regulatory_element.refers[0].nodes[0].id == 1622
    assert tmp_regulatory_element.refers[0].nodes[0].x == 993.28081
    assert tmp_regulatory_element.refers[0].nodes[0].y == 981.69366
    assert tmp_regulatory_element.refers[0].nodes[1].id == 1631
    assert tmp_regulatory_element.refers[0].nodes[1].x == 993.34466
    assert tmp_regulatory_element.refers[0].nodes[1].y == 981.02774
    assert tmp_regulatory_element.refers[0].nodes[2].id == 1625
    assert tmp_regulatory_element.refers[0].nodes[2].x == 993.40880
    assert tmp_regulatory_element.refers[0].nodes[2].y == 980.35882
    assert all(
        ref.type == WayType.TRAFFIC_SIGN
        if len(tmp_regulatory_element.refers) > 0
        else True
        for ref in tmp_regulatory_element.refers
    )
    assert (
        RegulatoryElement.deserialize(tmp_regulatory_element.serialize())
        == tmp_regulatory_element
    )
    assert isinstance(tmp_regulatory_element.to_geometry(), list) and all(
        isinstance(geom, LineString)
        for geom in tmp_regulatory_element.to_geometry()
    )
