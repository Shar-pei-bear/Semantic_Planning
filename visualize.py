import habitat_sim
from habitat.utils.visualizations import maps
import cv2
import numpy as np
from skimage.draw import line

def setup_path_visualization(path_follower, obj_attr_mgr, rigid_obj_mgr, vis_samples=100):
    vis_objs = []
    sphere_handle = obj_attr_mgr.get_template_handles("uvSphereSolid")[0]
    sphere_template_cpy = obj_attr_mgr.get_template_by_handle(sphere_handle)
    sphere_template_cpy.scale *= 0.2
    template_id = obj_attr_mgr.register_template(sphere_template_cpy, "mini-sphere")
    print("template_id = " + str(template_id))
    if template_id < 0:
        return None
    vis_objs.append(rigid_obj_mgr.add_object_by_template_handle(sphere_handle))

    for point in path_follower._points:
        cp_obj = rigid_obj_mgr.add_object_by_template_handle(sphere_handle)
        if cp_obj.object_id < 0:
            print(cp_obj.object_id)
            return None
        cp_obj.translation = point
        vis_objs.append(cp_obj)

    for i in range(vis_samples):
        cp_obj = rigid_obj_mgr.add_object_by_template_handle("mini-sphere")
        if cp_obj.object_id < 0:
            print(cp_obj.object_id)
            return None
        cp_obj.translation = path_follower.pos_at(float(i / vis_samples))
        vis_objs.append(cp_obj)

    for obj in vis_objs:
        if obj.object_id < 0:
            print(obj.object_id)
            return None

    for obj in vis_objs:
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    return vis_objs


def draw_objects(top_down_map, objects, fog_of_war, sim):
    for obj in objects:
        a_x, a_y = maps.to_grid(
            obj.translation[2],
            obj.translation[0],
            top_down_map.shape[0:2],
            sim=sim,
        )

        if fog_of_war[a_x, a_y]:
            cv2.circle(top_down_map, (a_y, a_x), radius=10, color=(255, 255, 0), thickness=-1)

    return top_down_map

def draw_target_objects(top_down_map, objects, sim):
    for obj in objects:
        a_x, a_y = maps.to_grid(
            obj.aabb.center[2],
            obj.aabb.center[0],
            top_down_map.shape[0:2],
            sim=sim,
        )

        cv2.drawMarker(top_down_map, (a_y, a_x), color=(0, 255, 255), markerType = cv2.MARKER_DIAMOND,
                       markerSize = 20, thickness=5, line_type=8)

    return top_down_map

def draw_start_positions(top_down_map, positions, sim):
    for i, position in enumerate(positions):
        a_x, a_y = maps.to_grid(
            position[2],
            position[0],
            top_down_map.shape[0:2],
            sim=sim,
        )

        cv2.drawMarker(top_down_map, (a_y, a_x), color=(255, 0, 0), markerType = cv2.MARKER_SQUARE,
                       markerSize = 20, thickness=1, line_type=8)
        cv2.putText(top_down_map, str(i+1), (a_y-7, a_x+8), color=(255, 0 ,0), fontFace = cv2.FONT_ITALIC, fontScale=0.7)

    return top_down_map


def find_sightings(top_down_map, obj, sim, img_id):
    pathfinder = sim.pathfinder

    lower_bound, upper_bound = pathfinder.get_bounds()

    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / top_down_map.shape[0],
        abs(upper_bound[0] - lower_bound[0]) / top_down_map.shape[1],
    )

    resolution = np.amin(grid_size)
    max_line_len = 3 / resolution
    angles = np.arange(
        -np.pi, np.pi, step=1.0 / max_line_len, dtype=np.float32
    )

    a_x, a_y = maps.to_grid(
        obj.aabb.center[2],
        obj.aabb.center[0],
        top_down_map.shape[0:2],
        sim=sim,
    )

    start = np.asarray([a_x, a_y], dtype=int)
    object_sighting = np.zeros_like(top_down_map, dtype=np.uint8)
    sighting_contour =  np.zeros_like(top_down_map, dtype=np.uint8)
    for angle in angles:
        end = start + max_line_len * np.array([np.cos(angle),
                                               np.sin(angle)])
        end = np.rint(end).astype(np.int)
        discrete_line = list(zip(*line(start[0], start[1],
                                       end[0], end[1])))

        for pt in discrete_line:
            x, y = pt

            if np.linalg.norm(start - np.asarray(pt), ord=1) < 50:
                continue

            if x < 0 or x >= top_down_map.shape[0]:
                break

            if y < 0 or y >= top_down_map.shape[1]:
                break

            if top_down_map[x, y] == 0:
                break

            distance = np.linalg.norm(start - np.asarray(pt)) * resolution
            if distance > 2:
                object_sighting[x, y] = 255

    kernel = np.ones((3, 3), np.uint8) * 255
    object_sighting = cv2.dilate(object_sighting, kernel, iterations=1)
    object_sighting = cv2.erode(object_sighting, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(object_sighting, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours

    # cv2.drawContours(sighting_contour, contours, -1, (255, 255, 255), 3)
    #
    # cv2.imwrite('debug' +str(img_id) +'.png', sighting_contour)


