# creature.py

import pygame
# from direction_nn import DirectionNet

class Creature:
    def __init__(self, x, y, speed, radius, color, name, direction='up'):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.direction = direction
        self.name = name
        self.color = color
        self.nearby_creatures = []
        self.ticks = 0
        self.action = 0
        # self.brain = DirectionNet()

    def move(self, direction):
        if direction == 'up':
            self.y -= self.speed
        elif direction == 'down':
            self.y += self.speed
        elif direction == 'left':
            self.x -= self.speed
        elif direction == 'right':
            self.x += self.speed

    def set_speed(self, speed):
        self.speed = speed

    def collision_check(self, other):
        distance = ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
        return distance < self.radius + other.radius
    
    def collision_move(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        distance = ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
        distance = max(distance, 0.1)
        self.x += dx / distance * self.speed
        self.y += dy / distance * self.speed
        other.x -= dx / distance * self.speed
        other.y -= dy / distance * self.speed

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)