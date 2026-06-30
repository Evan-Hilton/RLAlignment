import torch
import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import is_image_space

from src.evaluation.Button import Button
from configs.loaders.load_config import load_experiment_config
from src.evaluation.multiple_channel_obs_debug_vis import *

pygame.init()

# ------------------------------------------------ necessary game settings ------------------------------------------------------

game_name = "agent visualizer" # the name of the window that pops up
WIDTH = 1500 # pixels
HEIGHT = 900 # pixels
FRAME = 0
FRAME_RATE = 120 # frames / second
background_color = (0, 0, 0) # black
graph_color = (210, 240, 210) # nice light green color
centroid_detection_color1 = (252, 53, 213) # PINK!
centroid_detection_color2 = (17, 74, 77) # a good dark turqoise
text_color = (255, 255, 255) # white
button_inside_color = centroid_detection_color2
button_border_color = (255, 255, 255) # white

# ----------------------------------------------------- variables ---------------------------------------------------------------

#config = load_experiment_config("configs/experiments/d06_10_26_tubeDragging0.1NoImageMultPanel.yaml")
#config = load_experiment_config("configs/experiments/d06_10_26_tubeDragging0.5NoImageMultPanel.yaml")
#config = load_experiment_config("configs/experiments/d06_12_26_fineAlignmentNaiveConfig.yaml")
#config = load_experiment_config("configs/experiments/d06_09_26_noImage.yaml")
config = load_experiment_config("configs/experiments/d06_26_26_fineAlignmentStackedCnnWithAction.yaml")

env = config["environment"](config["env_params"])
telescope = env.telescope
img_scale = 4 * 128 / env.telescope.img_size

def load_model(path, env):
    model = PPO.load(path, env=env)
    return model

model = load_model(config["model_save_path"], env)
obs = env.reset()[0]
reward = []
done = False
feature_vector = None

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
    env.telescope.update()
    scaled = np.repeat(np.repeat(telescope.image, img_scale, axis=0), img_scale, axis=1)
    pixels = pygame.surfarray.pixels3d(surface)

    pixels[:, :, 0] = scaled.T
    pixels[:, :, 1] = scaled.T
    pixels[:, :, 2] = scaled.T

    del pixels
    # draw centroid locations
    if hasattr(env, "detected_centroids"): draw_centroids(surface, env.detected_centroids)
    else: draw_centroids(surface, env.telescope.true_centroids)

    # draw screen center
    if draw_screen_center:
        draw_centroids(surface, [env.telescope.center])

"""
    Adds a little symbol indicating where all of the detected
    centroids in the image are
"""
def draw_centroids(surface, centroids):
    global centroid_detection_color
    for centroid in centroids:
        x, y = env.telescope.fp_to_uv(centroid[0], centroid[1])
        draw_point_indicator(surface, centroid_detection_color1, centroid_detection_color2, 4, (x*img_scale, y*img_scale))

def draw_point_indicator(surface, inside_color, border_color, radius, location):
    pygame.draw.circle(surface, border_color, location, radius)
    pygame.draw.circle(surface, inside_color, location, radius - 2)

"""
    Renders a live view of diagnostics of the agent.
    screen is 895x515
"""
def render_agent_screen(surface):
    if is_image_space(env.observation_space):
        render_tiled_images(surface, obs)

"""
    Renders buttons and such.
    screen is 1112x292
"""
def render_UI_screen(surface):

    surface.fill(background_color)

    for button in buttons:
        button.draw(surface, font)

def reset_sim():
    global obs, reward, done
    reward = []
    obs = env.reset()[0]
    done = False
    compute_feature_vector(obs)

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
    if keys[pygame.K_SPACE]:
        single_step()

def single_step():
    global done
    if not done:
        advance()

"""
    does one frame step of the simulation by passing 
    the current state of the telescope to the agent,
    receiving an action and updating the telescope.
"""
def advance():
    global done, reward, obs, feature_vector

    action, _ = model.predict(obs, deterministic=True)

    obs, r, terminated, truncated, _ = env.step(action)
    reward.append(r)

    compute_feature_vector(obs)
    #print(np.min(feature_vector), np.max(feature_vector))

    done = terminated or truncated

def compute_feature_vector(observation):
    global feature_vector
    # Convert the observation to the format expected by the policy
    obs_tensor, _ = model.policy.obs_to_tensor(observation)

    # Compute the feature vector
    with torch.no_grad():
        feature_vector = model.policy.features_extractor(obs_tensor)

    # Remove the batch dimension
    feature_vector = feature_vector.squeeze(0).cpu().numpy()

"""
    draws an outline around the window and also labels it
"""
def outline_window(main_surface, sub_surface, sub_surface_location, name, font):
    sub_rect = sub_surface.get_rect()
    pygame.draw.rect(main_surface, text_color, (sub_surface_location[0] - 1, sub_surface_location[1] - 1, sub_rect.width + 2, sub_rect.height + 2), 1)

    # place the label at the bottom middle of the window
    text_surface = font.render(name, False, text_color)
    text_rect = text_surface.get_rect()
    x = sub_surface_location[0] + sub_rect.width * 0.5 - text_rect.width * 0.5
    y = sub_surface_location[1] + sub_rect.height + text_rect.height * 0.5
    main_surface.blit(text_surface, (x, y))

def render_feature_view(surface, graph_color):
    surface.fill(background_color)

    # surface dimensions and border padding
    surf_rect = (surface.get_rect()[2], surface.get_rect()[3])
    border_size = (0.05 * surf_rect[0], 0.05 * surf_rect[1])
    bar_length = surf_rect[0] - (2 * border_size[0])

    # x axis
    pygame.draw.line(surface, (255, 255, 255), (border_size[0], surf_rect[1] * 0.5), (surf_rect[0] - border_size[0], surf_rect[1] * 0.5))

    # bar graph
    bar_width = int(bar_length / len(feature_vector))
    if bar_width < 1: 
        print(f"WARNING: FEATURE VECTOR TOO BIG TO VISUALIZE. VECTOR SIZE {len(feature_vector)} TOO BIG")
        return
    
    max_value  = np.abs(np.max(feature_vector))
    max_height = (surf_rect[1] - (2 * border_size[1])) * 0.5
    for i, value in enumerate(feature_vector):
        x = i * bar_width
        h = (value / max_value) * max_height
        if h >= 0:
            pygame.draw.rect(surface, graph_color, (x + border_size[0], border_size[1] + max_height - h, bar_width - 1 if bar_width > 1 else 1, h))
        else:
            pygame.draw.rect(surface, graph_color, (x + border_size[0], border_size[1] + max_height, bar_width - 1 if bar_width > 1 else 1, -h))
    
    # values
    text_surface = font.render(str(int(max_value * 10)/10), False, text_color)
    text_rect = text_surface.get_rect()
    x = border_size[0] * 0.5 - text_rect.width * 0.5
    y = border_size[1] - text_rect.height * 0.5
    surface.blit(text_surface, (x, y))

# -------------------------------------------------- background functionality -------------------------------------------------

# main window
main_window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(game_name)
clock = pygame.time.Clock()
run = True
font = pygame.font.SysFont('Times New Roman', 15)
logo_visible = True # set to True to see logo in corner
draw_screen_center = True

buttons = [
    Button(
        rect=(20, 20, 120, 40),
        text="Reset",
        callback=reset_sim,
        button_inside_color=button_inside_color,
        button_border_color=button_border_color,
        text_color=text_color
    ),

    Button(
        rect=(20, 80, 120, 40),
        text="Step",
        callback=single_step,
        button_inside_color=button_inside_color,
        button_border_color=button_border_color,
        text_color=text_color
    )
]

if logo_visible: logo = pygame.image.load("src/evaluation/rosalina.png").convert_alpha()

# sub windows
telescope_view =    pygame.Surface((512, 512))  # at 30  , 30
telescope_location = (30, 30)
agent_view =        pygame.Surface((895, 514))  # at 573 , 30
agent_location = (573, 30)
ui_view =           pygame.Surface((160, 292))  # at 30  , 576
ui_location = (30, 576)
reward_view =       pygame.Surface((292, 292))  # at 1175, 576
reward_location = (1175, 576)
feature_view =      pygame.Surface((922, 292))  # at 220 , 576
feature_location = (220, 576)

reset_sim()

while run:
    for event in pygame.event.get():
        # handle closing the program
        if event.type == pygame.QUIT:
            run = False
        
        # handle button clicks
        for button in buttons:
            button.handle_event(
                event,
                ui_location
            )
    
    main_loop(FRAME) # the main computation loop

    # fill the background with the background color
    main_window.fill(background_color)
    if logo_visible: main_window.blit(logo, (30, 0))

    # draw an outline around every window
    outline_window(main_window, telescope_view, telescope_location, "telescope view", font)
    outline_window(main_window, agent_view, agent_location, "agent diagnostics", font)
    outline_window(main_window, ui_view, ui_location, "ui", font)
    outline_window(main_window, reward_view, reward_location, "reward vs time", font)
    outline_window(main_window, feature_view, feature_location, "feature vector", font)

    # render the windows themselves
    render_telescope_screen(telescope_view)
    render_agent_screen(agent_view)
    render_UI_screen(ui_view)
    render_reward_screen(reward_view, graph_color)
    render_feature_view(feature_view, graph_color)

    main_window.blit(telescope_view, telescope_location)
    main_window.blit(agent_view, agent_location)
    main_window.blit(ui_view, ui_location)
    main_window.blit(reward_view, reward_location)
    main_window.blit(feature_view, feature_location)

    input_loop(pygame.key.get_pressed(), pygame.mouse.get_pressed(), pygame.mouse.get_pos()) # a list of all inputs

    # increment the frame number. This is to keep track of what frame the game is on
    FRAME += 1
    clock.tick(FRAME_RATE) # keeps the game running at a max speed of 'FRAME_RATE' frames/second (might be slower if computations are heavy)
    # this makes sure everything is closed properly when the window is closed
    pygame.display.flip()

pygame.quit()