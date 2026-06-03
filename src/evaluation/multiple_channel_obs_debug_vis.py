import math
import numpy as np
import pygame

def render_agent_diag(surface, env):
    render_tiled_images(surface, env.get_observation(), scale=2)

def render_tiled_images(surface, images, scale=4, padding=8):
    n, H, W = images.shape

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        # Convert grayscale -> RGB
        rgb = np.repeat(img[:, :, None], 3, axis=2)

        img_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

        if scale != 1:
            img_surface = pygame.transform.scale(
                img_surface,
                (W * scale, H * scale)
            )

        x = col * (W * scale + padding)
        y = row * (H * scale + padding)

        surface.blit(img_surface, (x, y))