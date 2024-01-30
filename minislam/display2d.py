#!/usr/bin/env python3
import pygame


class Display:
    def __init__(self, width, height):
        self.width, self.height = width, height
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.surface = pygame.Surface(self.screen.get_size())

    def paint(self, img):
        for _ in pygame.event.get(): pass
        pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1))
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()