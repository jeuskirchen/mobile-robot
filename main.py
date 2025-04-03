import pygame
import math
import random
from generate_map import generate_map


# Constants
WIDTH, HEIGHT = 800, 800
ROBOT_RADIUS = 10
STEP_SIZE = 5 
SENSOR_LENGTH = 50 
NUM_SENSORS = 12
SENSOR_ANGLE_STEP = 360/NUM_SENSORS
BACKGROUND_COLOR = (30, 30, 30)
ROBOT_COLOR = (200, 50, 50)
SENSOR_COLOR = (50, 200, 50)
OBSTACLE_COLOR = (100, 100, 100)
SENSOR_HIT_COLOR = (230, 0, 0)
MOVE_SPEED = 3
ROTATE_SPEED = 3

# Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Initialize font for displaying text
pygame.font.init()
font = pygame.font.SysFont(None, 18)

# Robot position and orientation
robot_x, robot_y = WIDTH // 2, HEIGHT // 2
robot_angle = -90  # In degrees

# Motor speed  -------------------------------------------------------
v_l = 0  # Left motor speed
v_r = 0  # Right motor speed

obstacles = generate_map(WIDTH, HEIGHT, cell_size=50) 



def is_robot_colliding(x, y, obstacles):
    """Checks if the robot's position collides with any obstacles."""
    robot_rect = pygame.Rect(x - ROBOT_RADIUS, y - ROBOT_RADIUS, ROBOT_RADIUS * 2, ROBOT_RADIUS * 2)
    for obs in obstacles:
        if robot_rect.colliderect(obs):
            return True
    return False


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
                    break
            if hit:
                break

        sensors.append((sensor_x, sensor_y, hit, distance))
    return sensors




# Function to draw the robot and its sensors
# I edited this to draw the motor speed values inside the robot circle ----------------
def draw_robot(x, y, angle, sensors, v_l, v_r):
    # Draw sensors based on precomputed data
    for sensor_x, sensor_y, hit, _ in sensors:
        color = SENSOR_HIT_COLOR if hit else SENSOR_COLOR
        pygame.draw.line(screen, color, (x, y), (sensor_x, sensor_y), 2)

    # Draw the robot body (circle)
    pygame.draw.circle(screen, ROBOT_COLOR, (int(x), int(y)), ROBOT_RADIUS)

    # Draw the direction line (heading)
    heading_x = x + ROBOT_RADIUS * math.cos(math.radians(angle))
    heading_y = y + ROBOT_RADIUS * math.sin(math.radians(angle))
    pygame.draw.line(screen, (0, 0, 0), (x, y), (heading_x, heading_y), 3)

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
        screen.blit(distance_text, (text_x - 10, text_y - 10))  # Center the text


def draw_info(screen, x, y, angle, font):
    """Draw the robot's position and angle on the screen."""
    info_text = f"{int(angle)}, ({int(x)}, {int(y)})"
    text_surface = font.render(info_text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(bottomright=(WIDTH - 10, HEIGHT - 10))
    screen.blit(text_surface, text_rect)

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


# Makes sure the robot is not placed on top of or inside an obstacle
while is_robot_colliding(robot_x, robot_y, obstacles):
    # Reposition the robot randomly within the map bounds
    robot_x = random.randint(ROBOT_RADIUS, WIDTH - ROBOT_RADIUS)
    robot_y = random.randint(ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS)


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Reset motor speeds at the start of each frame -------------------------------
    v_l, v_r = 0, 0

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w] and keys[pygame.K_RIGHT]:
        v_l, v_r = MOVE_SPEED, MOVE_SPEED
    elif keys[pygame.K_w]:
        v_l, v_r = MOVE_SPEED, MOVE_SPEED - 1  # Left wheel faster → turn left
    elif keys[pygame.K_RIGHT]:
        v_l, v_r = MOVE_SPEED - 1, MOVE_SPEED  # Right wheel faster → turn right
    else:
        v_l, v_r = 0, 0

    # Move robot using motor speeds
    v = (v_r + v_l) / 2
    omega = (v_r - v_l) / 40  # wheel base hardcoded for now
    robot_angle += math.degrees(omega)
    dx = v * math.cos(math.radians(robot_angle))
    dy = v * math.sin(math.radians(robot_angle))
    new_x = robot_x + dx
    new_y = robot_y + dy

    if not is_robot_colliding(new_x, new_y, obstacles):
        robot_x, robot_y = new_x, new_y


    # Calculate sensors once per frame
    sensors = calculate_sensors(robot_x, robot_y, robot_angle)

    # Example: Print sensor data
    for i, (sx, sy, hit, distance) in enumerate(sensors):
        status = "HIT" if hit else "CLEAR"
        if hit: 
            print(f"Sensor {i + 1}: {status} at ({sx:.1f}, {sy:.1f}), distance {distance}")

    # Drawing
    screen.fill(BACKGROUND_COLOR)

    # Draw obstacles
    for obs in obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, obs)

    # Draw robot with sensors and info 
    draw_robot(robot_x, robot_y, robot_angle, sensors, v_l, v_r) # added motor speeds ---------------
    draw_info(screen, robot_x//STEP_SIZE, robot_y//STEP_SIZE, 360-robot_angle, font)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()