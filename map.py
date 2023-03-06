import numpy as np
MAP_THICKNESS_SCALAR: int = 128
from habitat.utils.visualizations import fog_of_war, maps
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
import cv2

class TopDownMap():
    r"""Top Down Map measure"""

    def __init__(
        self, sim: "HabitatSim", map_resolution, max_episode_steps, fov, visibility_dist, agent
    ):
        self._sim = sim
        self._agent = agent
        self._max_episode_steps = max_episode_steps
        self._step_count: Optional[int] = None
        self._map_resolution = map_resolution
        self._previous_xy_location: Optional[Tuple[int, int]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self._fov = fov
        self._visibility_dist = visibility_dist
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        print(self.line_thickness)
        self._fog_of_war_mask = None
        self._metric = None

    def get_original_map(self, agent_position):
        top_down_map = maps.get_topdown_map(
            self._sim.pathfinder, height=agent_position[1],
            map_resolution=self._map_resolution,
        )

        self._fog_of_war_mask = np.zeros_like(top_down_map)
        return top_down_map

    def return_original_map(self):
        return self._original_map

    def return_grid_coord(self):
        agent_position = self._agent.get_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        return [a_x, a_y]

    def reset_metric(self):
        self._step_count = 0
        self._metric = None
        agent_position = self._agent.get_state().position
        self._top_down_map = self.get_original_map(agent_position)
        self._original_map = self._top_down_map.copy()

        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)
        self.update_fog_of_war_mask(np.array([a_x, a_y]))

    def update_metric(self):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._agent.get_state().position
        )

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
        }

        return self._metric

    def get_polar_angle(self):
        agent_state = self._agent.get_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._max_episode_steps, 245
            )

            thickness = self.line_thickness
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
            self._top_down_map,
            self._fog_of_war_mask,
            agent_position,
            self.get_polar_angle(),
            fov=self._fov,
            max_line_len=self._visibility_dist
            / maps.calculate_meters_per_pixel(
                self._map_resolution, sim=self._sim
            ),
        )

    def get_metric(self):
        return self._metric
