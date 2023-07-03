"""INTERACTION track data API.

This module provides a set of tools for working with the INTERACTION track data
stored in the CSV files. The track data API parses the CSV files and provides a
convenient interface for accessing the track data.
"""
# Copyright (c) 2023, Juanwu Lu <juanwu@purdue.edu>.
# Released under the BSD-3-Clause license.
# See https://opensource.org/license/bsd-3-clause/ for licensing details.
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from .tracks.container import (
    MOTION_STATE_FIELD_MAPPING,
    INTERACTIONCase,
    MotionState,
    Track,
    TrackFrame,
)
from .tracks.typing import AgentType


class INTERACTIONScenarioLayers(IntEnum):
    """Enum for the layers in the INTERACTION scenario track data."""

    MOTION_STATE = 0
    """Data layer of all the agent motion states."""
    TRACK = 1
    """Data layer of all the agent tracks."""


class INTERACTIONScenario:
    """Track data API class for a single INTERACTION dataset scenario.

    Attributes:
        num_cases (int): the number of cases in the scenario.
        location (str): the string name of the scenario location.
        track_root (str): the root directory of the track data files.
    """

    __slots__ = (
        "_location",
        "_root",
        "_frames",
        "_split",
        "_tracks_to_predict",
        "_interesting_agents",
        "_num_cases",
    )

    def __init__(
        self, root: str, location: str, split: Optional[str] = None
    ) -> None:
        self._root = Path(root).resolve()
        self._location = location

        self._split = split
        if self._split is None:
            # dereference the split
            split = Path(self._root).parts[-1]
            if "train" in split:
                self._split = "train"
            elif "val" in split:
                self._split = "val"
            elif "test" in split:
                self._split = "test"
            else:
                raise ValueError(f"Unable to dereference split from {root}.")
        assert self._split in [
            "train",
            "val",
            "test",
        ], f"Invalid split {self._split}."

        self._init_scenario()

    @property
    def num_cases(self) -> int:
        """int: the number of cases in the scenario."""
        return self._num_cases

    @property
    def location(self) -> str:
        """str: the string name of the scenario location."""
        return self._location

    @property
    def track_root(self) -> str:
        """str: the root directory of the track data files."""
        return str(self._root)

    @property
    def track_file(self) -> str:
        """str: the path to the track data file."""
        if self._split == "test":
            filename = f"{self._location}_obs.csv"
        else:
            filename = f"{self._location}_{self._split}.csv"

        return str(Path(self._root).joinpath(filename))

    @property
    def tracks_to_predict(self) -> dict[int, list[int]]:
        """dict[int, list[int]]: the tracks to predict for each case."""
        return self._tracks_to_predict

    @property
    def interesting_agents(self) -> dict[int, list[int]]:
        """dict[int, list[int]]: the interesting agents for each case."""
        return self._interesting_agents

    def get_case(self, case_id: int) -> INTERACTIONCase:
        return INTERACTIONCase(
            location=self._location,
            case_id=int(case_id),
            history_frame=self._get_history(int(case_id)),
            current_frame=self._get_current(int(case_id)),
            futural_frame=self._get_futural(int(case_id)),
            tracks_to_predict=self._tracks_to_predict[int(case_id)],
            interesting_agents=self._interesting_agents[int(case_id)],
        )

    def render(
        self,
        case_id: int,
        anchor: Optional[tuple[float, float, float]] = None,
        ax: Optional[plt.Axes] = None,
        mode: str = "tail-box",
    ) -> plt.Axes:
        if case_id >= self._num_cases:
            raise ValueError(
                f"Invalid case_id {case_id} for scenario {self._location}."
            )

        return self.get_case(case_id).render(anchor=anchor, ax=ax, mode=mode)

    def _get_history(self, case_id: int) -> TrackFrame:
        df = self._frames[INTERACTIONScenarioLayers.MOTION_STATE].loc[case_id]
        return df.loc[df["timestamp_ms"] <= 1000]

    def _get_current(self, case_id: int) -> TrackFrame:
        df = self._frames[INTERACTIONScenarioLayers.MOTION_STATE].loc[case_id]
        return df.loc[df["timestamp_ms"] == 1000]

    def _get_futural(self, case_id: int) -> TrackFrame:
        df = self._frames[INTERACTIONScenarioLayers.MOTION_STATE].loc[case_id]
        return df.loc[df["timestamp_ms"] > 1000]

    def _init_scenario(self) -> None:
        motion_state_df = pd.read_csv(self.track_file, delimiter=",")
        motion_state_df["agent_type"] = motion_state_df["agent_type"].apply(
            lambda x: AgentType.deserialize(x)
        )
        self._num_cases = len(motion_state_df["case_id"].unique())

        # extract tracks to predict and interesting agents
        if self._split == "test":
            tracks_to_predict = motion_state_df.loc[
                motion_state_df["tracks_to_predict"] == 1
            ]
            self._tracks_to_predict: dict[int, list[int]] = (
                tracks_to_predict.groupby("case_id")
                .agg(track_id=("track_id", lambda x: x.unique().tolist()))
                .to_dict()["track_id"]
                .to_dict()
            )
            interesting_agents = motion_state_df.loc[
                motion_state_df["interesting_agent"] == 1
            ]
            self._interesting_agents: dict[int, list[int]] = (
                interesting_agents.groupby("case_id")
                .agg(track_id=("track_id", lambda x: x.unique().tolist()))
                .to_dict()["track_id"]
                .to_dict()
            )
        else:
            tracks_to_predict = motion_state_df.groupby(
                ["case_id", "track_id"]
            ).agg(
                min_timestamp=("timestamp_ms", "min"),
                max_timestamp=("timestamp_ms", "max"),
                agent_type=("agent_type", "first"),
            )
            tracks_to_predict = tracks_to_predict.loc[
                (tracks_to_predict["min_timestamp"] == 100)
                & (tracks_to_predict["max_timestamp"] == 4000)
                & (tracks_to_predict["agent_type"] == AgentType.CAR.value)
            ].reset_index(drop=False)
            self._tracks_to_predict: dict[int, list[int]] = (
                tracks_to_predict.groupby("case_id")
                .agg(list)["track_id"]
                .to_dict()
            )
            self._interesting_agents: dict[int, list[int]] = {
                case_id: [] for case_id in motion_state_df.case_id.unique()
            }

        # store the frames
        #
        # process the motion state dataframe
        geometry = gpd.points_from_xy(
            x=motion_state_df["x"], y=motion_state_df["y"]
        )
        motion_state_gdf = gpd.GeoDataFrame(motion_state_df, geometry=geometry)
        track_df = (
            motion_state_df.loc[
                :, ["case_id", "track_id", "timestamp_ms", "agent_type"]
            ]
            .sort_values(["case_id", "track_id", "timestamp_ms"])
            .groupby(["case_id", "track_id"])
            .first()
            .reset_index()
        )

        self._frames = {
            INTERACTIONScenarioLayers.MOTION_STATE: motion_state_gdf.set_index(
                ["case_id", "track_id"]
            ),
            INTERACTIONScenarioLayers.TRACK: track_df.set_index(
                ["case_id", "track_id"]
            ),
        }

    def __getitem__(self, case_id: int) -> INTERACTIONCase:
        return self.get_case(case_id)

    def __len__(self) -> int:
        return self.num_cases

    def __str__(self) -> str:
        return (
            f"<INTERACTIONScenario(location={self._location}"
            f"root={self._root}) at {hex(id(self))}>"
        )

    def __repr__(self) -> str:
        return str(self)
