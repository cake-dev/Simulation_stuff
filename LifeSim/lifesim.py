import pygame
import sys
import random
import torch
from creature import Creature
from direction_nn import DirectionNet

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 800
RADIUS = 10
SPEED = 4  # Set the speed of the creature
FPS = 30
NUM_CREATURES = 40  # Number of creatures
# COLLISION_COUNT = 0  # Number of collisions
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# frame counter to keep track of how many frames have passed for creature movement
frame_counter = 0

# Neural network
direction_net = DirectionNet()

# Optimizer
optimizer = torch.optim.Adam(direction_net.parameters(), lr=0.001)

# Create the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the creature's attributes
creatures = []
for i in range(NUM_CREATURES):
    # creature = {'x': random.randint(RADIUS, WIDTH - RADIUS), 'y': random.randint(RADIUS, HEIGHT - RADIUS)}
    creature = Creature(random.randint(RADIUS, WIDTH - RADIUS), random.randint(RADIUS, HEIGHT - RADIUS), SPEED, RADIUS, direction=random.choice(['up', 'down', 'left', 'right']), color=WHITE, name='Creature {}'.format(i))
    creatures.append(creature)

# Set up the player's attributes
# player = {'x': WIDTH // 2, 'y': HEIGHT // 2}
player = Creature(WIDTH // 2, HEIGHT // 2, SPEED, RADIUS, direction=random.choice(['up', 'down', 'left', 'right']), color=RED, name='Player')

# Create a clock object to help with timing
clock = pygame.time.Clock()

# Initialize the state and action for each creature
for creature in creatures:
    creature.state = torch.tensor([creature.x, creature.y, 0], dtype=torch.float)
    creature.action = 0


# Game loop
while True:
    clock.tick(FPS) 

    frame_counter += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get the current state of the keys for player movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player.move('up')
    if keys[pygame.K_DOWN]:
        player.move('down')
    if keys[pygame.K_LEFT]:
        player.move('left')
    if keys[pygame.K_RIGHT]:
        player.move('right')

    # Make sure the player doesn't leave the screen
    player.x = max(min(player.x, WIDTH - RADIUS), RADIUS)
    player.y = max(min(player.y, HEIGHT - RADIUS), RADIUS)

    # Update each creature
    for creature in creatures:
        # Update the creature's ticks +5 or -5 with 50/50 probability
        if random.random() < 0.50:
            creature.ticks += 5
            if random.random() < 0.10:
                creature.ticks += 10
        else:
            creature.ticks -= 5
            if random.random() < 0.10:
                creature.ticks -= 10

        # Make sure ticks is between 0 and 60
        creature.ticks = max(min(creature.ticks, FPS), 0)

        # Get the previous state and action
        prev_state = creature.state
        prev_action = creature.action
        nearby_creatures = len(creature.nearby_creatures)

        # Change direction
        # if creature.ticks >= FPS:
        creature.ticks = 0
        # Get the creature's current state
        state = torch.tensor([creature.x, creature.y, nearby_creatures], dtype=torch.float)
        # Use the model to get a probability distribution over the directions
        probs = direction_net(state)
        # Choose a direction based on the probabilities
        directions = ['up', 'down', 'left', 'right']
        # creature.action = random.choices(range(4), weights=probs.detach().numpy())[0]
        creature.action = random.choices(range(4), weights=probs.squeeze().detach().numpy())[0]
        creature.direction = directions[creature.action]
        # Randomly change the speed
        if random.random() < 0.10:
            creature.speed += random.randint(-1, 1)
            # print('{} speed changed to {}'.format(creature.name, creature.speed))
        frame_counter = 0
        # Calculate the reward
        reward = -nearby_creatures  # Reward is negative number of nearby creatures
        # Update the model
        optimizer.zero_grad()
        action_probs = direction_net(prev_state)
        loss = -torch.log(action_probs[0, prev_action]) * reward  # Negative log likelihood loss
        loss = loss.mean()  # Take the mean of the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(direction_net.parameters(), max_norm=1)
        optimizer.step()
        # Make sure speed is greater than 0
        creature.speed = max(creature.speed, 0)
        creature.move(creature.direction)
        # Make sure the creature doesn't leave the screen
        creature.x = max(min(creature.x, WIDTH - RADIUS), RADIUS)
        creature.y = max(min(creature.y, HEIGHT - RADIUS), RADIUS)
        # if creature.ticks >= FPS:
        #     creature.ticks = 0
        #     creature.direction = random.choice(['up', 'down', 'left', 'right'])
        #     # Randomly change the speed
        #     if random.random() < 0.10:
        #         creature.speed += random.randint(-1, 1)
        #         print('{} speed changed to {}'.format(creature.name, creature.speed))
        #     frame_counter = 0

        # # Make sure speed is greater than 0
        # creature.speed = max(creature.speed, 0)

        # creature.move(creature.direction)

        # # Make sure the creature doesn't leave the screen
        # creature.x = max(min(creature.x, WIDTH - RADIUS), RADIUS)
        # creature.y = max(min(creature.y, HEIGHT - RADIUS), RADIUS)

        # Check for collisions with other creatures and count nearby creatures
        creature.nearby_creatures = []
        for creature2 in creatures:
            if creature != creature2:
                distance = ((creature.x - creature2.x) ** 2 + (creature.y - creature2.y) ** 2) ** 0.5
                if distance < 100:
                    creature.nearby_creatures.append(creature2)
                    # print('{} near {}'.format(creature.name, creature2.name))
                if creature.collision_check(creature2):
                    creature.collision_move(creature2)
                    # print('Collision: {}'.format(COLLISION_COUNT))
                    # COLLISION_COUNT += 1

    # Check for collisions with the player and count nearby creatures
    for creature in creatures:
        distance = ((player.x - creature.x) ** 2 + (player.y - creature.y) ** 2) ** 0.5
        if distance < 100:
            player.nearby_creatures.append(creature)
            print('Player near {}'.format(creature.name))
        if creature.collision_check(player):
            creature.collision_move(player)
            player.collision_move(creature)
            # print('Collision: {}'.format(COLLISION_COUNT))
            # COLLISION_COUNT += 1

    # Draw everything
    screen.fill((0, 0, 0))

    # Draw the creatures
    for creature in creatures:
        creature.draw(screen)

    # Drw the player
    player.draw(screen)
        
    # Flip the display
    pygame.display.flip()