import numpy as np

P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]
P2s = [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128]

"""
    When a panel gets rotated, the tubes that structurally connect
    panels get dragged along with the panel, and rotates the other
    panels in the vicinity of the panel that moved. Given a panel
    in the primary mirror of the pSCT, this function returns a list
    of panels and rotations to make to those panels to simulate
    tube dragging.

    primary_panel_id: the id of the primary panel [ex: 1112]
    rotation: how much to rotate the primary panel [ex: np.array([0.5, -0.1])]. The rotation is normalized [-1, 1]
    effect_strength: a value in [0-1]. a value of 0 means no effect, a value of 1 means the other panels rotate with the same amount as 'rotation'

    returns: a dictionary with panel ids and rotations to make to each
"""
def drag_tubes(primary_panel_id, rotation, effect_strength):
    # if you stand at the center of the mirror ring and look at primary_panel, then left panel is the panel directly to the left, and similarly defined for right
    panel_actions = {}

    left_panel_rotation = [x * effect_strength for x in rotation] # scale rotation by effect strength
    left_panel_rotation[0] *= -1
    right_panel_rotation = left_panel_rotation

    primary_in_P1 = primary_panel_id in P1s
    if primary_in_P1:
        left_panel_id = P1s[(P1s.index(primary_panel_id) - 1) % len(P1s)]
        right_panel_id = P1s[(P1s.index(primary_panel_id) + 1) % len(P1s)]
    else: # primary in P2
        left_panel_id = P2s[(P2s.index(primary_panel_id) - 1) % len(P2s)]
        right_panel_id = P2s[(P2s.index(primary_panel_id) + 1) % len(P2s)]
    
    panel_actions[left_panel_id] = left_panel_rotation
    panel_actions[right_panel_id] = right_panel_rotation

    return panel_actions

"""
    Adds a random amount of noise to the rotation signal.
    rotation should be a list with size 2: the x, y rotation to make.
    two random numbers between -noise and noise will be chosen and added to
    each value in rotation
"""
def add_rotation_noise(rotation, noise):
    noise_x = np.random.uniform(-noise, noise)
    noise_y = np.random.uniform(-noise, noise)
    rotation[0] += noise_x
    rotation[1] += noise_y