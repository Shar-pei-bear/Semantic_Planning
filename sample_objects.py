import habitat_sim
import numpy as np
import magnum as mn
import random
import math

# sample a random valid state for the object from the scene bounding box or navmesh
def sample_object_state(
    sim, obj, from_navmesh=True, maintain_object_up=True, max_tries=100, bb=None, stage_attr_mgr=None
):
    # check that the object is not STATIC
    if obj.motion_type is habitat_sim.physics.MotionType.STATIC:
        print("sample_object_state : Object is STATIC, aborting.")
    if from_navmesh:
        if not sim.pathfinder.is_loaded:
            print("sample_object_state : No pathfinder, aborting.")
            return False
    elif not bb:
        print(
            "sample_object_state : from_navmesh not specified and no bounding box provided, aborting."
        )
        return False
    tries = 0
    valid_placement = False
    # Note: following assumes sim was not reconfigured without close
    scene_collision_margin = stage_attr_mgr.get_template_by_id(0).margin
    while not valid_placement and tries < max_tries:
        tries += 1
        # initialize sample location to random point in scene bounding box
        sample_location = np.array([0, 0, 0])
        if from_navmesh:
            # query random navigable point
            sample_location = sim.pathfinder.get_random_navigable_point()
        else:
            sample_location = np.random.uniform(bb.min, bb.max)
        # set the test state
        obj.translation = sample_location
        if maintain_object_up:
            # random rotation only on the Y axis
            y_rotation = mn.Quaternion.rotation(
                mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
            )
            obj.rotation = y_rotation * obj.rotation
        else:
            # unconstrained random rotation
            obj.rotation = ut.random_quaternion()

        # raise object such that lowest bounding box corner is above the navmesh sample point.
        if from_navmesh:
            obj_node = obj.root_scene_node
            xform_bb = habitat_sim.geo.get_transformed_bb(
                obj_node.cumulative_bb, obj_node.transformation
            )
            # also account for collision margin of the scene
            obj.translation += mn.Vector3(
                0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
            )

        # test for penetration with the environment
        if not sim.contact_test(obj.object_id):
            valid_placement = True

    if not valid_placement:
        return False
    return True
