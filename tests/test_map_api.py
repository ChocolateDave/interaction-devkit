from pathlib import Path

import pytest

from interaction.dataset.map_api import INTERACTIONMap, INTERACTIONMapLayers
from interaction.dataset.maps.elements import (
    Lanelet,
    MultiPolygon,
    Node,
    RegulatoryElement,
    Way,
)


@pytest.fixture(scope="module")
def test_create_map_api():
    root = Path(__file__).parents[1].joinpath(Path("data/maps"))
    if not root.exists():
        pytest.skip("No map data available!")

    # Test loading
    try:
        map_api = INTERACTIONMap(root=root, location="DR_USA_Roundabout_FT")
    except Exception as e:
        pytest.fail(f"Failed to load map: {e}")

    return map_api


@pytest.mark.usefixtures("test_create_map_api")
def test_map_api_properties(test_create_map_api: INTERACTIONMap):
    # Test properties
    assert test_create_map_api.bounds == pytest.approx(
        (
            956.714470999199,
            963.1091317205809,
            1073.5684297027765,
            1036.8814364117022,
        ),
        abs=1e-2,
    )
    assert test_create_map_api.map_root == str(
        Path(__file__).parents[1].joinpath(Path("data/maps")).resolve()
    )
    assert test_create_map_api.location == "DR_USA_Roundabout_FT"


@pytest.mark.usefixtures("test_create_map_api")
def test_map_api_getters(test_create_map_api: INTERACTIONMap):
    # Test get_available_layers
    try:
        layer_dict = test_create_map_api.get_available_layers()
    except Exception as e:
        pytest.fail(f"Failed to get available layers: {e}")
    assert all(
        (
            isinstance(layer, INTERACTIONMapLayers)
            and isinstance(num_objects, int)
            and num_objects >= 0
        )
        for layer, num_objects in layer_dict.items()
    )

    # Test get_map_layer
    try:
        test_create_map_api.get_map_layer("lanelet")
    except Exception as e:
        pytest.fail(f"Failed to get map layer: {e}")

    # Test get_map_object
    try:
        obj = test_create_map_api.get_map_object(1001, "node")
        assert isinstance(obj, Node), f"Expected a `Node`, but got {type(obj)}"
    except Exception as e:
        pytest.fail(f"Failed to get map object: {e}")

    try:
        obj = test_create_map_api.get_map_object(10001, "way")
        assert isinstance(obj, Way), f"Expected a `Way`, but got {type(obj)}"
    except Exception as e:
        pytest.fail(f"Failed to get map object: {e}")

    try:
        obj = test_create_map_api.get_map_object(30001, "lanelet")
        assert isinstance(
            obj, Lanelet
        ), f"Expected a `Lanelet`, but got {type(obj)}"
    except Exception as e:
        pytest.fail(f"Failed to get map object: {e}")

    try:
        obj = test_create_map_api.get_map_object(40001, "multipolygon")
        assert isinstance(
            obj, MultiPolygon
        ), f"Expected a `MultiPolygon`, but got {type(obj)}"
    except Exception as e:
        pytest.fail(f"Failed to get map object: {e}")

    try:
        obj = test_create_map_api.get_map_object(50001, "regulatory_element")
        assert isinstance(
            obj, RegulatoryElement
        ), f"Expected a `RegulatoryElement`, but got {type(obj)}"
    except Exception as e:
        pytest.fail(f"Failed to get map object: {e}")

    with pytest.raises(KeyError):
        test_create_map_api.get_map_object(10001, "node")


@pytest.mark.skip(reason="No test cases available")
def test_map_api_get_proximal():
    pass


@pytest.mark.usefixtures("test_create_map_api")
def test_map_api_render(test_create_map_api: INTERACTIONMap):
    # Test render without specifying anchor
    try:
        test_create_map_api.render()
    except Exception as e:
        pytest.fail(f"Failed to render map: {e}")

    # Test render with specifying anchor
    try:
        test_create_map_api.render(anchor=(0.0, 0.0, 0.0))
    except Exception as e:
        pytest.fail(f"Failed to render map: {e}")

    # Test render with specifying anchor and radius
    try:
        test_create_map_api.render(anchor=(0.0, 0.0, 0.0), radius=100)
    except Exception as e:
        pytest.fail(f"Failed to render map: {e}")
