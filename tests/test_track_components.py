import math
from copy import deepcopy

import pytest

from interaction.dataset.tracks.container import (
    INTERACTIONCase,
    MotionState,
    Track,
)
from interaction.dataset.tracks.typing import AgentType


# ----------- Test typing ----------- #
def test_agent_type() -> None:
    # test the enum values
    assert AgentType.UNDEFINED == 0
    assert AgentType.CAR == 1
    assert AgentType.PEDESTRIAN_BICYCLE == 2

    # test the enum names
    assert AgentType(0).name == "UNDEFINED"
    assert AgentType(1).name == "CAR"
    assert AgentType(2).name == "PEDESTRIAN_BICYCLE"

    # test deserialize function
    assert AgentType.deserialize("UNDEFINED") == AgentType.UNDEFINED
    assert AgentType.deserialize("CAR") == AgentType.CAR
    assert (
        AgentType.deserialize("PEDESTRIAN_BICYCLE")
        == AgentType.PEDESTRIAN_BICYCLE
    )
    assert AgentType.deserialize("undefined") == AgentType.UNDEFINED
    assert AgentType.deserialize("car") == AgentType.CAR
    assert (
        AgentType.deserialize("pedestrian_bicycle")
        == AgentType.PEDESTRIAN_BICYCLE
    )
    assert AgentType.deserialize(0) == AgentType.UNDEFINED
    assert AgentType.deserialize(1) == AgentType.CAR
    assert AgentType.deserialize(2) == AgentType.PEDESTRIAN_BICYCLE

    # test one-hot encoding
    assert AgentType.one_hot_serialize(AgentType.UNDEFINED) == [0, 0]
    assert AgentType.one_hot_serialize(AgentType.CAR) == [1, 0]
    assert AgentType.one_hot_serialize(AgentType.PEDESTRIAN_BICYCLE) == [0, 1]


# ----------- Test container ----------- #
@pytest.fixture(scope="module")
def test_create_motion_state() -> None:
    # test the constructor
    tmp_motion_state = MotionState(
        agent_id=1,
        timestamp_ms=100,
        position_x=0.0,
        position_y=0.0,
        velocity_x=-3.0,
        velocity_y=4.0,
        heading=math.pi / 4,
        length=5.0,
        width=2.0,
    )
    assert tmp_motion_state.agent_id == 1
    assert tmp_motion_state.timestamp_ms == 100
    assert tmp_motion_state.position_x == 0.0
    assert tmp_motion_state.position_y == 0.0
    assert tmp_motion_state.velocity_x == -3.0
    assert tmp_motion_state.velocity_y == 4.0
    assert tmp_motion_state.heading == math.pi / 4
    assert tmp_motion_state.length == 5.0
    assert tmp_motion_state.width == 2.0

    return tmp_motion_state


@pytest.mark.skip(reason="TODO")
@pytest.mark.usefixtures("test_create_motion_state")
def test_state_properties(test_create_motion_state: MotionState) -> None:
    # test motion state properties
    groud_truth_box = [
        (1.69, 2.77),
        (2.77, 1.69),
        (5.23, 6.31),
        (6.31, 5.23),
    ]
    coords = [
        (round(coord[0], 2), round(coord[1], 2))
        for coord in test_create_motion_state.bounding_box.exterior.coords
    ]
    assert coords[:-1] == pytest.approx(groud_truth_box, abs=1e-2)
    assert test_create_motion_state.speed == pytest.approx(5.0, abs=1e-2)


@pytest.mark.usefixtures("test_create_motion_state")
def test_state_to_geometry(test_create_motion_state: MotionState) -> None:
    # test the to_geometry method
    assert test_create_motion_state.to_geometry().wkt == "POINT (0 0)"


@pytest.mark.usefixtures("test_create_motion_state")
def test_state_comparisons(test_create_motion_state: MotionState) -> None:
    # test the compare method
    assert test_create_motion_state == deepcopy(test_create_motion_state)
    other_motion_state = MotionState(
        agent_id=2,
        timestamp_ms=100,
        position_x=0.0,
        position_y=0.0,
        velocity_x=-3.0,
        velocity_y=4.0,
        heading=math.pi / 4,
        length=5.0,
        width=2.0,
    )
    assert test_create_motion_state != other_motion_state
    other_motion_state = MotionState(
        agent_id=1,
        timestamp_ms=200,
        position_x=0.0,
        position_y=0.0,
        velocity_x=-3.0,
        velocity_y=4.0,
        heading=math.pi / 4,
        length=5.0,
        width=2.0,
    )
    assert test_create_motion_state < other_motion_state
    assert test_create_motion_state <= other_motion_state
    assert other_motion_state > test_create_motion_state
    assert other_motion_state >= test_create_motion_state


@pytest.mark.skip(reason="TODO")
def test_track() -> None:
    pass
