import habitat_sim
import magnum as mn
# grip/release and sync gripped object state kineamtically
class ObjectGripper:
    def __init__(
        self,
        sim,
        agent_scene_node,
        end_effector_offset,
    ):
        self._sim = sim
        self._node = agent_scene_node
        self._offset = end_effector_offset
        self._gripped_obj = None
        self._gripped_obj_buffer = 0  # bounding box y dimension offset of the offset

    def sync_states(self):
        if self._gripped_obj is not None:
            agent_t = self._node.absolute_transformation_matrix()
            agent_t.translation += self._offset + mn.Vector3(
                0, self._gripped_obj_buffer, 0.0
            )
            self._gripped_obj.transformation = agent_t

    def grip(self, obj):
        if self._gripped_obj is not None:
            #print("Oops, can't carry more than one item.")
            return
        self._gripped_obj = obj
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        object_node = obj.root_scene_node
        self._gripped_obj_buffer = object_node.cumulative_bb.size_y() / 2.0
        self.sync_states()

    def release(self):
        if self._gripped_obj is None:
            #print("Oops, can't release nothing.")
            return
        self._gripped_obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        self._gripped_obj.linear_velocity = (
            self._node.absolute_transformation_matrix().transform_vector(
                mn.Vector3(0, 0, -1.0)
            )
            + mn.Vector3(0, 2.0, 0)
        )
        self._gripped_obj = None
