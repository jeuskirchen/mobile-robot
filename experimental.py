# TODO: refactor 
import pygame
import math
import random
import numpy as np 
# import datetime as dt 


# Constants
DRAW_OBSTACLES = False 
WIDTH, HEIGHT = 800, 800
ROBOT_RADIUS = 10
STEP_SIZE = 5 
SENSOR_LENGTH = 100 
NUM_SENSORS = 12
SENSOR_ANGLE_STEP = 360/NUM_SENSORS
MOVE_SPEED = 3
ROTATE_SPEED = 3
AUTOMATIC = False 
OBSTACLE_SIZE = 50
BACKGROUND_COLOR = (30, 30, 30)
ROBOT_COLOR = (200, 50, 50)
SENSOR_COLOR = (50, 200, 50)
OBSTACLE_COLOR = (100, 100, 100)
SENSOR_HIT_COLOR = (230, 0, 0)
ROBOT_POSITION_COLOR = (60, 60, 60) 

# Initialize PyGame
pygame.init()
# game_id = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Robot")

# Initialize font for displaying text
pygame.font.init()
font = pygame.font.SysFont(None, 18)

# Sets to store sensor hit points, robot position points and occupancy points
# TODO: instead of sets, probably better to use a 2D numpy array to keep track of 
#  each cell's occupancy, etc.
hit_points = set()
pos_points = set()
occupancy_points = set()  

# Initialize visibility flags
obstacles_visible = False 
minimap_visible = False
trajectory_visible = False  
occupancy_visible = True
hits_visible = True 
sensors_visible = True
belief_visible = False 


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
        rect = pygame.Rect(x, y, cell_size, cell_size)
        # Avoid overlapping obstacles
        if not any(rect.colliderect(obstacle) for obstacle in obstacles):
            obstacles.append(rect)
    # Add boundary walls to fully enclose the map
    obstacles.extend([
        pygame.Rect(0, 0, width, 5),          # Top boundary
        pygame.Rect(0, 0, 5, height),         # Left boundary
        pygame.Rect(0, height - 5, width, 5), # Bottom boundary
        pygame.Rect(width - 5, 0, 5, height)  # Right boundary
    ])
    return obstacles


# Generate random map with obstacles
obstacles = generate_map(WIDTH, HEIGHT, cell_size=OBSTACLE_SIZE) 


def is_robot_colliding(x, y, obstacles):
    """
    Checks if the robot's position collides with any obstacles.
    """
    robot_rect = pygame.Rect(x - ROBOT_RADIUS, y - ROBOT_RADIUS, ROBOT_RADIUS * 2, ROBOT_RADIUS * 2)
    for obs in obstacles:
        if robot_rect.colliderect(obs):
            return True
    return False


def place_robot(obstacles):
    robot_x, robot_y = WIDTH//2, HEIGHT//2
    robot_angle = -90  # In degrees 
    # Makes sure the robot is not placed on top of or inside an obstacle
    while is_robot_colliding(robot_x, robot_y, obstacles):
        # Reposition the robot randomly within the map bounds
        robot_x = random.randint(ROBOT_RADIUS, WIDTH - ROBOT_RADIUS)
        robot_y = random.randint(ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS)
    return robot_x, robot_y, robot_angle


def discretize_point(point, cell_size):
    """
    Project/discretize/quantize a point to closest grid point 
    """
    return (cell_size*(int(point[0] // cell_size)), cell_size*int(point[1] // cell_size))


def calculate_sensors(x, y, angle):
    sensors = []
    for i in range(NUM_SENSORS):
        sensor_angle = math.radians(angle + i * SENSOR_ANGLE_STEP)
        hit = False
        distance = float("inf")  
        sensor_x, sensor_y = x, y 
        # Step along the line to detect the closest intersection
        for d in range(0, SENSOR_LENGTH, STEP_SIZE):  
            sensor_x = x + d * math.cos(sensor_angle)
            sensor_y = y + d * math.sin(sensor_angle)
            point = (int(sensor_x), int(sensor_y))
            # Check collision with obstacles
            for obs in obstacles:
                if obs.collidepoint(point):
                    hit = True
                    # Store the distance where collision occurs, in "steps":
                    distance = max((d-ROBOT_RADIUS)//STEP_SIZE, 0)
                    # Add the hit point to the global list
                    # if point not in hit_points: 
                    #     hit_points.add(point)
                    anchor_point = discretize_point(point, STEP_SIZE) 
                    if anchor_point not in hit_points:
                        hit_points.add(anchor_point)
                    break
            if hit:
                break
            else:
                # Add the discretized point's to the occupancy area if no hit
                anchor_point = discretize_point(point, STEP_SIZE)
                if anchor_point not in occupancy_points:
                    occupancy_points.add(anchor_point)
        sensors.append((sensor_x, sensor_y, hit, distance))
    return sensors


# Function to move the robot along its heading
def move_robot(x, y, angle, speed):
    # Calculate new position based on current angle
    dx = speed * math.cos(math.radians(angle))
    dy = speed * math.sin(math.radians(angle))

    # Calculate the new position
    new_x = x + dx
    new_y = y + dy

    # Collision check (basic)
    robot_rect = pygame.Rect(new_x - ROBOT_RADIUS, new_y - ROBOT_RADIUS, ROBOT_RADIUS * 2, ROBOT_RADIUS * 2)
    for obs in obstacles:
        if robot_rect.colliderect(obs):
            return x, y  # Prevent movement if collision detected

    return new_x, new_y


def draw_robot(x, y, angle, sensors, v_l, v_r):
    # Draw the robot body (circle)
    pygame.draw.circle(screen, ROBOT_COLOR, (int(x), int(y)), ROBOT_RADIUS)

    # Draw the direction line (heading)
    heading_x = x + ROBOT_RADIUS * math.cos(math.radians(angle))
    heading_y = y + ROBOT_RADIUS * math.sin(math.radians(angle))
    pygame.draw.line(screen, (0, 0, 0), (x, y), (heading_x, heading_y), 3)

    if sensors_visible:
        # Draw sensors based on precomputed data
        for sensor_x, sensor_y, hit, _ in sensors:
            color = SENSOR_HIT_COLOR if hit else SENSOR_COLOR
            pygame.draw.line(screen, color, (x, y), (sensor_x, sensor_y), 2)

        # Draw motor speed values (text)
        v_l_text = font.render(f"L: {v_l}", True, (255, 255, 255))
        v_r_text = font.render(f"R: {v_r}", True, (255, 255, 255))

        # Display motor speed text inside the robot body
        screen.blit(v_l_text, (x - 20, y - 10))  # Adjust position for left motor speed
        screen.blit(v_r_text, (x - 20, y + 10))  # Adjust position for right motor speed

        # Draw sensor distance values around the robot (text)
        for i, (_, _, hit, distance) in enumerate(sensors):
            sensor_angle = math.radians(angle + i * SENSOR_ANGLE_STEP)
            text_x = x + (ROBOT_RADIUS + 20) * math.cos(sensor_angle)  # Position outside the robot
            text_y = y + (ROBOT_RADIUS + 20) * math.sin(sensor_angle)

            # Handle infinite distance
            distance_display = "∞" if distance == float("inf") else f"{int(distance)}"
            distance_text = font.render(distance_display, True, (255, 255, 255) if hit else (100, 100, 100))
            # Get the rectangle of the text and center it
            text_rect = distance_text.get_rect(center=(text_x, text_y))
            screen.blit(distance_text, text_rect)


def draw_info(screen, x, y, angle, font):
    """
    Draw the robot's position and angle on the screen.
    """
    info_text = f"{int(angle)}, ({int(x)}, {int(y)})"
    text_surface = font.render(info_text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(bottomright=(WIDTH - 10, HEIGHT - 10))
    screen.blit(text_surface, text_rect)


def draw_minimap(screen, hit_points, occupancy_points, pos_points, belief, robot_x, robot_y, robot_angle):
    """
    Draw a minimap in the top-right corner of the screen.
    """

    MINIMAP_SCALE = 0.2  # Scale factor for the minimap
    MINIMAP_WIDTH = int(WIDTH * MINIMAP_SCALE)
    MINIMAP_HEIGHT = int(HEIGHT * MINIMAP_SCALE)
    MINIMAP_MARGIN = 10  # Margin from the screen edges
    MINIMAP_X = WIDTH - MINIMAP_WIDTH - MINIMAP_MARGIN
    MINIMAP_Y = MINIMAP_MARGIN

    ALPHA = 255  # 150

    # Create a transparent surface for the minimap
    minimap_surface = pygame.Surface((MINIMAP_WIDTH, MINIMAP_HEIGHT), pygame.SRCALPHA)
    minimap_surface.fill((50, 50, 50, ALPHA))

    # Draw explored points on the minimap surface
    if occupancy_visible:
        for point in occupancy_points:
            minimap_point = (
                int(point[0] * MINIMAP_SCALE),
                int(point[1] * MINIMAP_SCALE),
            )
            pygame.draw.circle(minimap_surface, (50, 50, 200, ALPHA), minimap_point, 1)  # Blue dots for explored area

    # Draw hit points on the minimap surface
    if hits_visible:
        for point in hit_points:
            minimap_point = (
                int(point[0] * MINIMAP_SCALE),
                int(point[1] * MINIMAP_SCALE),
            )
            pygame.draw.circle(minimap_surface, (*SENSOR_HIT_COLOR, ALPHA), minimap_point, 1)  # Red dots for hit points

    if trajectory_visible:
        for point in pos_points:
            minimap_point = (
                MINIMAP_X + int(point[0] * MINIMAP_SCALE),
                MINIMAP_Y + int(point[1] * MINIMAP_SCALE),
            )
            pygame.draw.circle(screen, (*ROBOT_POSITION_COLOR, ALPHA), minimap_point, 1)  # Gray dots for robot positions

    # Draw the robot's current position and heading on the minimap surface
    robot_minimap_x = int(robot_x * MINIMAP_SCALE)
    robot_minimap_y = int(robot_y * MINIMAP_SCALE)
    pygame.draw.circle(minimap_surface, ROBOT_COLOR + (200,), (robot_minimap_x, robot_minimap_y), int(ROBOT_RADIUS * MINIMAP_SCALE))

    # Draw the robot's heading line on the minimap surface
    heading_x = robot_minimap_x + int(ROBOT_RADIUS * MINIMAP_SCALE * math.cos(math.radians(robot_angle)))
    heading_y = robot_minimap_y + int(ROBOT_RADIUS * MINIMAP_SCALE * math.sin(math.radians(robot_angle)))
    pygame.draw.line(minimap_surface, (255, 255, 255, ALPHA), (robot_minimap_x, robot_minimap_y), (heading_x, heading_y), 1)

    # Blit the minimap surface onto the main screen
    screen.blit(minimap_surface, (MINIMAP_X, MINIMAP_Y))


if __name__ == "__main__":
    # Robot position, orientation and initial motor speeds 
    robot_x, robot_y, robot_angle = place_robot(obstacles)
    last_action = pygame.K_w 
    best_direction = 0 
    # TODO: Keep track of robot's belief about its state: 
    initial_x, initial_y, initial_angle = robot_x, robot_y, robot_angle 
    # The belief state is formulated as a delta to the initial state:
    # only position for now; orientation later 
    belief = np.ones((WIDTH//STEP_SIZE, HEIGHT//STEP_SIZE))
    belief /= belief.sum()

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_ESCAPE:
                        running = False
                    case pygame.K_m:
                        minimap_visible = not minimap_visible
                    case pygame.K_t:
                        trajectory_visible = not trajectory_visible
                    case pygame.K_o:
                        occupancy_visible = not occupancy_visible
                    case pygame.K_h:
                        hits_visible = not hits_visible
                    case pygame.K_b:
                        # Unused right now
                        belief_visible = not belief_visible
                    case pygame.K_s:
                        sensors_visible = not sensors_visible
                    case pygame.K_RETURN: 
                        AUTOMATIC = not AUTOMATIC
                    case pygame.K_SPACE:
                        obstacles_visible = not obstacles_visible

        # Reset motor speeds at the start of each frame
        v_l, v_r = 0, 0

        # Add the robot's current position to the position history
        if (robot_x, robot_y) not in pos_points: 
            # pos_points.add((robot_x, robot_y)) 
            anchor_point = discretize_point((robot_x, robot_y), STEP_SIZE)
            pos_points.add(anchor_point) 

        # Calculate sensors once per frame
        sensors = calculate_sensors(robot_x, robot_y, robot_angle)

        if AUTOMATIC:
            """
            # Policy 1:
            key = np.random.choice(actions, p=[0.5, 0.25, 0.25])
            if np.random.uniform() < 0.5:
                key = last_action
                last_action = key
            """
            # Policy 2: 
            # Determine best angle based on sensor data, then move in that direction
            # But certain percentage of the time, pick a random direction (randomly last direction +1, -1 or +0)
            exploration_rate = 0.2
            current_direction = best_direction
            if sum([hit for _, _, hit, _ in sensors]) == 0: 
                # If no sensor detects a hit, just keep going in current direction 
                best_direction = current_direction
            elif np.random.uniform() < exploration_rate:
                # best_direction = np.random.choice(range(NUM_SENSORS))
                # Pick random direction close to the last best direction
                best_direction = (current_direction + np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])) % NUM_SENSORS 
                # TODO: only pick random direction if that direction is not blocked
                # TODO: explicitly keep track of "frontiers" 
                #   Perhaps also color those frontier points differently, so it's clearer 
                #
                #
            else:
                # Choose the best direction based on sensor data
                best_direction = np.argmax([distance for _, _, hit, distance in sensors])
            robot_angle = (robot_angle + best_direction * SENSOR_ANGLE_STEP) % 360 
            robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, MOVE_SPEED)
            v_l, v_r = MOVE_SPEED, MOVE_SPEED  # Both motors move forward at the same speed
        else:
            # Handle key presses for movement and rotation
            keys = pygame.key.get_pressed()

            # Movement logic 
            if keys[pygame.K_w] and keys[pygame.K_LEFT]:
                # Move forward while turning left
                robot_angle = (robot_angle - ROTATE_SPEED) % 360  # Adjust angle for turning
                robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, MOVE_SPEED)
                v_l, v_r = MOVE_SPEED - 1, MOVE_SPEED  # Left motor slower than right motor
            elif keys[pygame.K_w] and keys[pygame.K_RIGHT]:
                # Move forward while turning right
                robot_angle = (robot_angle + ROTATE_SPEED) % 360  # Adjust angle for turning
                robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, MOVE_SPEED)
                v_l, v_r = MOVE_SPEED, MOVE_SPEED - 1  # Right motor slower than left motor
            elif keys[pygame.K_w]:
                # Move forward
                robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, MOVE_SPEED)
                v_l, v_r = MOVE_SPEED, MOVE_SPEED  # Both motors move forward at the same speed
            elif keys[pygame.K_LEFT]:
                # Rotate in place counterclockwise
                robot_angle = (robot_angle - ROTATE_SPEED) % 360
                v_l, v_r = -ROTATE_SPEED, ROTATE_SPEED  # Left motor backward, right motor forward
            elif keys[pygame.K_RIGHT]:
                # Rotate in place clockwise
                robot_angle = (robot_angle + ROTATE_SPEED) % 360
                v_l, v_r = ROTATE_SPEED, -ROTATE_SPEED  # Right motor backward, left motor forward
            else:
                # No movement
                v_l, v_r = 0, 0 

        # Drawing
        screen.fill(BACKGROUND_COLOR)

        # Draw obstacles
        if obstacles_visible:
            for obs in obstacles:
                pygame.draw.rect(screen, OBSTACLE_COLOR, obs)

        # Draw explored area
        if occupancy_visible:
            for point in occupancy_points:
                pygame.draw.circle(screen, (50, 50, 200), point, 1)  # Blue dots for occupancy area

        # Draw permanent hit points
        if hits_visible:
            for point in hit_points:
                pygame.draw.circle(screen, SENSOR_HIT_COLOR, point, 3)  # Draw a small red dot
        
        # Draw permanent positions points
        if trajectory_visible:
            for point in pos_points:
                pygame.draw.circle(screen, ROBOT_POSITION_COLOR, point, 3)  # Draw a small _ dot

        # Draw robot with sensors and info
        draw_robot(robot_x, robot_y, robot_angle, sensors, v_l, v_r)
        # draw_info(screen, robot_x // STEP_SIZE, robot_y // STEP_SIZE, 360 - robot_angle, font)

        # Toggle occupancy overlay
        if minimap_visible:
            draw_minimap(screen, hit_points, occupancy_points, pos_points, belief, robot_x, robot_y, robot_angle) 

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
