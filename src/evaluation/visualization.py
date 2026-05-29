import pygame
import numpy as np
from stable_baselines3 import PPO
from environment_old import pSCT_environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
pygame.init()

# ------------------------------------------------ necessary game settings ------------------------------------------------------

game_name = "agent evaluation visualization" # the name of the window that pops up
WIDTH = 1352 # pixels
HEIGHT = 484 + 50 # pixels
FRAME = 0
FRAME_RATE = 60 # frames / second
background_color = (0, 0, 0) # rgb color; each value ranges from 0-225 inclusive

# ----------------------------------------------------- variables ---------------------------------------------------------------

# Load environment (must match training env)
version = "3.6"
env = make_vec_env(
        pSCT_environment,
        n_envs=1
        #vec_env_cls=SubprocVecEnv, # recommended in the documentation for speeding up training
    )
env = VecNormalize.load("/Users/evanhilton/Desktop/VSCoding/RLAlignIndividual/envs/ppo_mirrorENV_v" + version, env)
env.training = False
env.norm_reward = False
env1 = env.envs[0]
base_env = env.envs[0].unwrapped

# Load trained model
model = PPO.load("/Users/evanhilton/Desktop/VSCoding/RLAlignIndividual/models/ppo_mirror_v" + version)

obs, _ = env1.reset()

# ----------------------------------------------------- game logic --------------------------------------------------------------

img_size = base_env.img_size # number of pixels wide
pix_size = 3 # how many pixels wide each pixel in the observation is rendered as
buffer = 50
play = False
rwd = []
show_center = True

# any external variable used should be declared global at the start of the function
def main_loop(FRAME): # the current frame number is passed to the main loop if you so choose to use it
    global play, obs, rwd
    if play:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env1.step(action)
        rwd.append(reward)
        print(reward)
        if terminated or truncated:
            play = False

def paint_loop(screen):
    global play, obs, rwd
    # image
    font = pygame.font.SysFont('Times New Roman', 18)
    text = "image"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (buffer + img_size * pix_size / 2 - font.size(text)[0] / 2, buffer / 2))

    pygame.draw.rect(screen, (100, 225, 50), (buffer + pix_size * 128 / 2, buffer + pix_size * 128 / 2, pix_size, pix_size)) # image center
    pygame.draw.rect(screen, (255, 255, 255), (buffer - 1, buffer - 1, pix_size * img_size + 2, pix_size * img_size + 2), width = 2) # bounding box of image
    for r in range(img_size):
        for c in range(img_size):
            col = 255 * obs[0][r][c]
            pygame.draw.rect(screen, (col, col, col), (c * pix_size + buffer, r * pix_size + buffer, pix_size, pix_size)) # image
    

    # detected
    text = "detected centroids"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (buffer + img_size * pix_size + buffer + img_size * pix_size / 2 - font.size(text)[0] / 2, buffer / 2))

    det = base_env._detected_fp_coords(obs[0])
    pygame.draw.rect(screen, (255, 255, 255), (buffer + pix_size * img_size + buffer - 1, buffer - 1, pix_size * img_size + 2, pix_size * img_size + 2), width = 2) # bounding box of detected
    for (fx, fy) in (det):
        u, v = base_env._fp_to_uv(fx, fy)
        x, y = u * pix_size + (buffer + pix_size * img_size + buffer), v * pix_size + buffer
        pygame.draw.rect(screen, (255, 255, 255), (x, y, pix_size, pix_size), width = 2) # bounding box of detected
    

    # reward graph
    text = "reward vs time"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (buffer + 2 * img_size * pix_size + buffer + buffer + img_size * pix_size / 2 - font.size(text)[0] / 2, buffer / 2))
    x0 = buffer + pix_size * img_size + buffer + pix_size * img_size + buffer # these are the screen pixel coordinates of the origin of the graph
    y0 = pix_size * img_size + buffer
    graph_size = pix_size * img_size
    # axes
    pygame.draw.line(screen, (255, 255, 255), (x0, y0), (x0, buffer))
    pygame.draw.line(screen, (255, 255, 255), (x0, y0), (x0 + graph_size, y0))
    # values
    n_points = len(rwd)
    if n_points > 0:
        dx = graph_size / n_points
        max_graph_val = max(rwd)
        min_graph_val = min(rwd)
        _range = max_graph_val - min_graph_val
        scale = (graph_size / _range) if _range != 0 else 1# this is in pixels / (reward units)
        prev = (x0, y0)
        for i, reward in enumerate(rwd):
            x = x0 + dx * i
            y = y0 - (reward - min_graph_val) * scale
            pygame.draw.line(screen, (255, 255, 255), prev, (x, y))
            prev = (x, y)
    
    # center
    if show_center:
        centerx, centery = base_env._fp_to_uv(base_env.center[0], base_env.center[1])
        xi, yi = centerx * pix_size + buffer, centery * pix_size + buffer
        xd, yd = centerx * pix_size + (buffer + pix_size * img_size + buffer), centery * pix_size + buffer
        pygame.draw.rect(screen, (100, 235, 50), (xi, yi, pix_size, pix_size), width = 2) # center of the image
        pygame.draw.rect(screen, (100, 235, 50), (xd, yd, pix_size, pix_size), width = 2) # center of the detected plane

def input_loop(keys, mouse, mouse_pos):
    global play, obs, rwd
    if not play and keys[pygame.K_SPACE]:
        play = True
    if keys[pygame.K_r]:
        play = False
        obs, _ = env1.reset()
        rwd = []


# -------------------------------------------------- background functionality -------------------------------------------------

# below makes the code function. It is not necessary to fully understand how it works
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(game_name)
clock = pygame.time.Clock()
run = True
while run:
    # fill the background with the background color
    screen.fill(background_color)

    main_loop(FRAME) # the current frame is passed to the main loop if you so choose to use it
    paint_loop(screen) # screen gets passed so that the print_loop function can draw on the screen
    input_loop(pygame.key.get_pressed(), pygame.mouse.get_pressed(), pygame.mouse.get_pos()) # a list of all inputs
    # the mouse_pressed input is a tuple containing: (left_click, middle_click, right_click)

    # increment the frame number. This is to keep track of what frame the game is on
    FRAME += 1
    clock.tick(FRAME_RATE) # keeps the game running at a max speed of 'FRAME_RATE' frames/second (might be slower if computations are heavy)
    # this makes sure everything is closed properly when the window is closed
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()