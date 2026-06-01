import pygame
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from configs.experiments.d06_01_26_onePanelBasicImageConfig import config

pygame.init()

# ------------------------------------------------ necessary game settings ------------------------------------------------------

game_name = "agent visualizer" # the name of the window that pops up
WIDTH = 1500 # pixels
HEIGHT = 900 # pixels
FRAME = 0
FRAME_RATE = 60 # frames / second
background_color = (0, 0, 0) # rgb color; each value ranges from 0-225 inclusive

# ----------------------------------------------------- variables ---------------------------------------------------------------

env = config["environment"](config["env_params"])
telescope = env.telescope

kwargs = {"env_config": config["env_params"]}
env = make_vec_env(
    config["environment"],
    n_envs=config["n_training_envs"],
    vec_env_cls=config["vec_env_cls"],
    env_kwargs=kwargs
)

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
    scaled = np.repeat(np.repeat(telescope.image, 4, axis=0), 4, axis=1)
    pixels = pygame.surfarray.pixels3d(surface)

    pixels[:, :, 0] = scaled.T
    pixels[:, :, 1] = scaled.T
    pixels[:, :, 2] = scaled.T

    del pixels

"""
    Renders a live view of diagnostics of the agent.
    screen is 895x515
"""
def render_agent_screen(surface):
    ...

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
def render_reward_screen(surface):
    ...

def input_loop(keys, mouse, mouse_pos):
    if keys[pygame.K_SPACE]:
        advance()

def advance():
    ...


# -------------------------------------------------- background functionality -------------------------------------------------

# main window
main_window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(game_name)
clock = pygame.time.Clock()
run = True

# sub windows
telescope_view =    pygame.Surface((512, 512))  # at 30  , 30
agent_view =        pygame.Surface((895, 515))  # at 573 , 30
ui_view =           pygame.Surface((1112, 292)) # at 30  , 576
reward_view =       pygame.Surface((292, 292))  # at 1175, 576

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the background with the background color
    main_window.fill(background_color)

    main_loop(FRAME) # the main computation loop

    render_telescope_screen(telescope_view)
    render_agent_screen(agent_view)
    render_UI_screen(ui_view)
    render_reward_screen(reward_view)

    main_window.blit(telescope_view, (30, 30))
    main_window.blit(agent_view, (573, 30))
    main_window.blit(ui_view, (30, 576))
    main_window.blit(reward_view, (1175, 576))

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