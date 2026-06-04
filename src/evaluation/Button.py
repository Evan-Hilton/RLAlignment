import pygame

class Button:

    def __init__(self, rect, text, callback, button_inside_color, button_border_color, text_color):

        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.button_inside_color = button_inside_color
        self.button_border_color = button_border_color
        self.text_color = text_color

    def draw(self, surface, font):

        pygame.draw.rect(surface, self.button_inside_color, self.rect)
        pygame.draw.rect(surface, self.button_border_color, self.rect, 2)

        text_surface = font.render(
            self.text,
            True,
            self.text_color
        )

        text_rect = text_surface.get_rect(
            center=self.rect.center
        )

        surface.blit(text_surface, text_rect)

    def handle_event(
        self,
        event,
        parent_surface_location
    ):

        if event.type != pygame.MOUSEBUTTONDOWN:
            return

        print("mouse down")

        if event.button != 1:
            return

        local_mouse = (
            event.pos[0] - parent_surface_location[0],
            event.pos[1] - parent_surface_location[1]
        )

        print("local mouse:", local_mouse)
        print("button rect:", self.rect)

        if self.rect.collidepoint(local_mouse):

            print("button clicked")

            self.callback()