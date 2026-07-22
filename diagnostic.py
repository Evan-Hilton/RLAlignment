import pygame
import numpy as np
from stable_baselines3 import PPO

from src.telescope.pSCT_P12 import PSCT_P12
from src.telescope.image_analyzer import ImageAnalyzer
from src.environments.no_image_psctp12_env import NoImagePSCTP12Env
from src.environments.fine_alignment.naive_image_env import NaiveImageEnv

pygame.init()

"""
    This python file is used for testing graphics related code. For example,
    it might be used to test the centroid fitter for generated telescope images.
"""

# ------------------------------------------------ necessary game settings ------------------------------------------------------

game_name = "agent visualizer" # the name of the window that pops up
WIDTH = 1500 # pixels
HEIGHT = 900 # pixels
FRAME = 0
FRAME_RATE = 60 # frames / second
background_color = (0, 0, 0)
graph_color = (210, 240, 210)
centroid_detection_color1 = (252, 53, 213)
centroid_detection_color2 = (17, 74, 77)

# ----------------------------------------------------- variables ---------------------------------------------------------------

#env = config["environment"](config["env_params"])
# telescope = PSCT_P1({
#     "img_size": 128,

#     "sigma_r_center": 0.75,
#     "sigma_r_max": 3.85,
#     "sigma_theta_center": 0.75,
#     "sigma_theta_max": 0.9,
#     "centroid_type": "gaussian",

#     "n_panels": 2,
#     "center_fp": np.array([1612.2804, 1024.4423]),

#     "init_scatter_pix": 500.0,
#     "init_rxry_scale": 0.05,
#     "img_fov_pix": 600.0,

#     "action_scale": 0.1,

#     "bg_level": 6,
#     "read_noise": 11
# })
tele = "configs/telescopes/fine_alignment/10_panel_img_noise/3.yaml"
#tele = "configs/telescopes/example.yaml"
env = NaiveImageEnv({
    "max_steps": 512,
    "telescope": tele
})
env.telescope.reset()
img_scale = 4 * 128 / env.telescope.img_size
det_cet = None

# ----------------------------------------------------- game logic --------------------------------------------------------------

def main_loop(FRAME): 
   global det_cet
   det_cet = ImageAnalyzer._sep_detection(env.telescope.image, {"threshold_sigma": 2.5,
                                                                "minarea": 20,
                                                                "deblend_nthresh": 8,
                                                                "deblend_cont": 0.05,
                                                                })
   columns = [
        "X_IMAGE",
        "Y_IMAGE",
        "FLUX_ISO",
        "FLUX_MAX",
        "BACKGROUND",
        "A_IMAGE",
        "B_IMAGE",
        "THETA_IMAGE",
        ]
   detected = det_cet[columns].to_numpy()
   obs = np.zeros((env.telescope.n_panels, 8), dtype=np.float32)
   n_detected = len(detected)
   obs[:n_detected] = detected[:n_detected]

   # normalization
   obs[:, 0] /= env.telescope.img_size
   obs[:, 1] /= env.telescope.img_size

   obs[:, 2] /= 5e+04

   obs[:, 3] /= 255
   obs[:, 4] /= 255

   obs[:, 5] /= 5
   obs[:, 6] /= 5

   obs[:, 7] /= 90

"""
    Renders the current live view of what the telescope sees.
    Also optionally displays the true centroid locations, center
    of the screen, and centroid fitting.
    screen is 512x512
"""
def render_telescope_screen(surface):
    scaled = np.repeat(np.repeat(env.telescope.image, img_scale, axis=0), img_scale, axis=1)
    pixels = pygame.surfarray.pixels3d(surface)

    pixels[:, :, 0] = scaled.T
    pixels[:, :, 1] = scaled.T
    pixels[:, :, 2] = scaled.T

    del pixels

    draw_detected_centroids(surface)

"""
    Adds a little symbol indicating where all of the detected
    centroids in the image are
"""
def draw_detected_centroids(surface):
    global centroid_detection_color
    for ID in det_cet['ID'].tolist():
        x, y = det_cet['X_IMAGE'][ID], det_cet['Y_IMAGE'][ID]
        A, B = det_cet['A_IMAGE'][ID], det_cet['B_IMAGE'][ID]
        height, width = 2*B, 2*A
        angle = -det_cet['THETA_IMAGE'][ID]
        draw_point_indicator(surface, centroid_detection_color1, centroid_detection_color2, 4, (x*img_scale, y*img_scale))
        draw_rotated_ellipse(surface, centroid_detection_color1, (x*img_scale, y*img_scale), width*img_scale, height*img_scale, angle, 2)

def draw_point_indicator(surface, inside_color, border_color, radius, location):
    pygame.draw.circle(surface, border_color, location, radius)
    pygame.draw.circle(surface, inside_color, location, radius - 2)

def draw_rotated_ellipse(surface, color, center, width, height, angle, outline_width):
    ellipse_surf = pygame.Surface((width, height), pygame.SRCALPHA)

    pygame.draw.ellipse(ellipse_surf, color, (0, 0, width, height), outline_width)

    rotated_surf = pygame.transform.rotate(ellipse_surf, angle)

    rotated_rect = rotated_surf.get_rect(center=center)

    surface.blit(rotated_surf, rotated_rect)

"""
    Renders buttons and such.
    screen is 1112x292
"""
def render_UI_screen(surface):
    ...

"""
    draws an outline around the window and also labels it
"""
def outline_window(main_surface, sub_surface, sub_surface_location, color, name, font):
    sub_rect = sub_surface.get_rect()
    pygame.draw.rect(main_surface, color, (sub_surface_location[0] - 1, sub_surface_location[1] - 1, sub_rect.width + 2, sub_rect.height + 2), 1)

    # place the label at the bottom middle of the window
    text_surface = font.render(name, False, color)
    text_rect = text_surface.get_rect()
    x = sub_surface_location[0] + sub_rect.width * 0.5 - text_rect.width * 0.5
    y = sub_surface_location[1] + sub_rect.height + text_rect.height * 0.5
    main_surface.blit(text_surface, (x, y))

prev_keys = None
def input_loop(keys, mouse, mouse_pos):
    global done, reward, prev_keys
    if keys[pygame.K_LEFT]:
        action = (1, 0)
        env.telescope.rotate_panel(1111, [action[0], action[1]])
        env.telescope.update()
    if keys[pygame.K_RIGHT]:
        action = (-1, 0)
        env.telescope.rotate_panel(1111, [action[0], action[1]])
        env.telescope.update()
    if keys[pygame.K_UP]:
        action = (0, -1)
        env.telescope.rotate_panel(1111, [action[0], action[1]])
        env.telescope.update()
    if keys[pygame.K_DOWN]:
        action = (0, 1)
        env.telescope.rotate_panel(1111, [action[0], action[1]])
        env.telescope.update()
    prev_keys = keys
    if keys[pygame.K_y]:
        env.telescope.reset()
    if keys[pygame.K_l]:
        env.telescope.true_centroids[0] = env.telescope.center
        env.telescope.update()
    if keys[pygame.K_r]:
        env.reset()
    if keys[pygame.K_b]:
        env.telescope.true_centroids = np.array([[1494.37987882, 875.5200608],[1652.49663262, 1140.64258532],[1745.1680854, 887.48300033],[1501.9680002, 1113.0027051]])
        env.telescope.update()

def advance(action):
    global done
    obs, _, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

# -------------------------------------------------- background functionality -------------------------------------------------

# main window
main_window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(game_name)
clock = pygame.time.Clock()
run = True
font = pygame.font.SysFont('Times New Roman', 15)

# sub windows
telescope_view =    pygame.Surface((512, 512))  # at 30  , 30
telescope_location = (30, 30)
ui_view =           pygame.Surface((1112, 292)) # at 30  , 576
ui_location = (30, 576)

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    input_loop(pygame.key.get_pressed(), pygame.mouse.get_pressed(), pygame.mouse.get_pos()) # a list of all inputs
    
    main_loop(FRAME) # the main computation loop

    # fill the background with the background color
    main_window.fill(background_color)

    # draw an outline around every window
    outline_window(main_window, telescope_view, telescope_location, (255, 255, 255), "telescope view", font)
    outline_window(main_window, ui_view, ui_location, (255, 255, 255), "ui", font)

    # render the windows themselves
    render_telescope_screen(telescope_view)
    render_UI_screen(ui_view)

    main_window.blit(telescope_view, telescope_location)
    main_window.blit(ui_view, ui_location)

    # increment the frame number. This is to keep track of what frame the game is on
    FRAME += 1
    clock.tick(FRAME_RATE) # keeps the game running at a max speed of 'FRAME_RATE' frames/second (might be slower if computations are heavy)
    # this makes sure everything is closed properly when the window is closed
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()