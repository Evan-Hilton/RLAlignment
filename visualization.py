import pygame
import numpy as np
from stable_baselines3 import PPO

from configs.experiments.d06_01_26_onePanelBasicImageConfig import config

pygame.init()

# ------------------------------------------------ necessary game settings ------------------------------------------------------

game_name = "agent visualizer" # the name of the window that pops up
WIDTH = 1500 # pixels
HEIGHT = 900 # pixels
FRAME = 0
FRAME_RATE = 60 # frames / second
background_color = (0, 0, 0) # black
graph_color = (210, 240, 210) # nice light green color
centroid_detection_color1 = (252, 53, 213) # PINK!
centroid_detection_color2 = (17, 74, 77) # a good dark turqoise

# ----------------------------------------------------- variables ---------------------------------------------------------------

env = config["environment"](config["env_params"])
telescope = env.telescope
img_scale = 4

def load_model(path, env):
    model = PPO.load(path, env=env)
    return model

model = load_model(config["model_save_path"], env)
obs = env.reset()[0]
reward = []
done = False

# ----------------------------------------------------- game logic --------------------------------------------------------------

def main_loop(FRAME): 
   ...

"""
    Renders the current live view of what the telescope sees.
    Also optionally displays the true centroid locations, center
    of the screen, and centroid fitting.
    screen is 512x512
"""
def render_telescope_screen(surface):
    scaled = np.repeat(np.repeat(telescope.image, img_scale, axis=0), img_scale, axis=1)
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
    for centroid in env.detected_centroids:
        x, y = env.telescope.fp_to_uv(centroid[0], centroid[1])
        pygame.draw.circle(surface, centroid_detection_color2, (x*img_scale, y*img_scale), 4)
        pygame.draw.circle(surface, centroid_detection_color1, (x*img_scale, y*img_scale), 2)

"""
    Renders a live view of diagnostics of the agent.
    screen is 895x515
"""
def render_agent_screen(surface):
    if "diagnostic_vis" not in config:
        return
    config["diagnostic_vis"](surface, env)

"""
    Renders buttons and such.
    screen is 1112x292
"""
def render_UI_screen(surface):
    ...

"""
    Renders the reward vs time graph.
    screen is 292x292
"""
def render_reward_screen(surface, graph_color):
    surface.fill(background_color)
    global reward
    if len(reward) < 2 or np.max(reward) - np.min(reward) == 0:
        return
    width, height = surface.get_rect().width, surface.get_rect().height
    graph_y_values = (reward - np.min(reward)) / (np.max(reward) - np.min(reward)) # normalize reward values to 0-1
    graph_y_values = height * graph_y_values # scale reward values to 0-surface height

    n = len(graph_y_values)
    points = [
        (
            i * (width - 1) / (n - 1),
            height - graph_y_values[i]  # flip y-axis
        )
        for i in range(n)
    ]

    pygame.draw.lines(surface, graph_color, False, points, 2)

def input_loop(keys, mouse, mouse_pos):
    global done, reward
    if keys[pygame.K_SPACE] and not done:
        advance()

"""
    does one frame step of the simulation by passing 
    the current state of the telescope to the agent,
    receiving an action and updating the telescope.
"""
def advance():
    global done, reward, obs
    action, _ = model.predict(obs, deterministic=True)

    obs, r, terminated, truncated, _ = env.step(action)
    reward.append(r)

    done = terminated or truncated

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
agent_view =        pygame.Surface((895, 514))  # at 573 , 30
agent_location = (573, 30)
ui_view =           pygame.Surface((1112, 292)) # at 30  , 576
ui_location = (30, 576)
reward_view =       pygame.Surface((292, 292))  # at 1175, 576
reward_location = (1175, 576)

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    main_loop(FRAME) # the main computation loop

    # fill the background with the background color
    main_window.fill(background_color)

    # draw an outline around every window
    outline_window(main_window, telescope_view, telescope_location, (255, 255, 255), "telescope view", font)
    outline_window(main_window, agent_view, agent_location, (255, 255, 255), "agent diagnostics", font)
    outline_window(main_window, ui_view, ui_location, (255, 255, 255), "ui", font)
    outline_window(main_window, reward_view, reward_location, (255, 255, 255), "reward vs time", font)

    # render the windows themselves
    render_telescope_screen(telescope_view)
    render_agent_screen(agent_view)
    render_UI_screen(ui_view)
    render_reward_screen(reward_view, graph_color)

    main_window.blit(telescope_view, telescope_location)
    main_window.blit(agent_view, agent_location)
    main_window.blit(ui_view, ui_location)
    main_window.blit(reward_view, reward_location)

    input_loop(pygame.key.get_pressed(), pygame.mouse.get_pressed(), pygame.mouse.get_pos()) # a list of all inputs

    # increment the frame number. This is to keep track of what frame the game is on
    FRAME += 1
    clock.tick(FRAME_RATE) # keeps the game running at a max speed of 'FRAME_RATE' frames/second (might be slower if computations are heavy)
    # this makes sure everything is closed properly when the window is closed
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()