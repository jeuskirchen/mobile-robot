import random
from pygame import Rect 

def generate_map(width, height, cell_size=10, density=0.2):
    """
    Generates a map with randomly placed blocks.
    - density: Fraction of the map area to be covered with obstacles (0.0 to 1.0).
    """
    obstacles = []

    # Calculate the maximum number of obstacles based on density
    max_obstacles = int((width * height) / (cell_size * cell_size) * density)

    for _ in range(max_obstacles):
        x = random.randint(0, (width // cell_size) - 1) * cell_size
        y = random.randint(0, (height // cell_size) - 1) * cell_size

        # Avoid placing obstacles too close to the starting position
        if (x < 100 and y < 100):
            continue

        # Create the obstacle rectangle
        rect = Rect(x, y, cell_size, cell_size)

        # Avoid overlapping obstacles
        if not any(rect.colliderect(obstacle) for obstacle in obstacles):
            obstacles.append(rect)

    # Add boundary walls to fully enclose the map
    obstacles.extend([
        Rect(0, 0, width, 5),          # Top boundary
        Rect(0, 0, 5, height),         # Left boundary
        Rect(0, height - 5, width, 5), # Bottom boundary
        Rect(width - 5, 0, 5, height)  # Right boundary
    ])

    return obstacles
