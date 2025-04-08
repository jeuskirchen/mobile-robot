import os 
import pygame
import math
import random
import numpy as np 
from generate_map import generate_map
from PIL import Image 
import imageio.v2 as imageio 
import datetime as dt 


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
SCREENCAPTURE = True 

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

last_frame = None 
frames = []



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
        # Get the rectangle of the text and center it
        text_rect = distance_text.get_rect(center=(text_x, text_y))
        screen.blit(distance_text, text_rect)


def draw_info(screen, x, y, angle, font):
    """Draw the robot's position and angle on the screen."""
    info_text = f"{int(angle)}, ({int(x)}, {int(y)})"
    text_surface = font.render(info_text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(bottomright=(WIDTH - 10, HEIGHT - 10))
    screen.blit(text_surface, text_rect)

# Function to move the robot along its heading
# This funciotn now handles collisions with obstacles and allows sliding along walls
# and also checks for collisions with the map boundaries--------------------------
# (all the sliding handeling and what not is contained only in this move_robot funciton)
def move_robot(x, y, angle, speed):
    # Calculate new position based on current angle
    dx = speed * math.cos(math.radians(angle))
    dy = speed * math.sin(math.radians(angle))

    # Calculate the new position
    new_x = x + dx
    new_y = y + dy

    # Collision check
    robot_rect = pygame.Rect(new_x - ROBOT_RADIUS, new_y - ROBOT_RADIUS, ROBOT_RADIUS * 2, ROBOT_RADIUS * 2)
    x_clear = True
    y_clear = True

    for obs in obstacles:
        if robot_rect.colliderect(obs):
            # Check for sliding along the wall
            # Try moving only in the x direction
            temp_x = x + dx
            temp_rect_x = pygame.Rect(temp_x - ROBOT_RADIUS, y - ROBOT_RADIUS, ROBOT_RADIUS * 2, ROBOT_RADIUS * 2)
            if temp_rect_x.colliderect(obs):
                x_clear = False

            # Try moving only in the y direction
            temp_y = y + dy
            temp_rect_y = pygame.Rect(x - ROBOT_RADIUS, temp_y - ROBOT_RADIUS, ROBOT_RADIUS * 2, ROBOT_RADIUS * 2)
            if temp_rect_y.colliderect(obs):
                y_clear = False

    # Resolve movement based on collision checks
    if not x_clear and not y_clear:
        # Block movement completely if both directions are blocked
        return x, y
    elif not x_clear:
        # Allow sliding along the y-axis
        return x, y + dy
    elif not y_clear:
        # Allow sliding along the x-axis
        return x + dx, y

    # If no collision, move to the new position
    return new_x, new_y


def save_frame(): 
    global frames, last_frame 
    frame_surface = pygame.display.get_surface().copy()
    frame_array3d = pygame.surfarray.array3d(frame_surface) 
    # Only add if frame is not empty and different from last frame 
    if (frame_array3d.sum() > 0) and (last_frame is None or (frame_array3d != last_frame).any()):
        frame_array = pygame.image.tostring(frame_surface, "RGB")
        frame = Image.frombytes("RGB", frame_surface.get_size(), frame_array)
        frames.append(frame) 
        last_frame = frame_array3d


def to_video():
    global frames
    if len(frames) == 0: 
        return
    if not os.path.exists("screencapture"):
        os.mkdir("screencapture")
    timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"screencapture/main {timestamp}.mp4"
    print(f"Saving file as {filename}") 
    with imageio.get_writer(filename, fps=60, codec="libx264") as writer: 
        for frame in frames[1:]:
            writer.append_data(np.array(frame))



# Makes sure the robot is not placed on top of or inside an obstacle
while is_robot_colliding(robot_x, robot_y, obstacles):
    # Reposition the robot randomly within the map bounds
    robot_x = random.randint(ROBOT_RADIUS, WIDTH - ROBOT_RADIUS)
    robot_y = random.randint(ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS)


# Main loop
running = True
while running:
    for event in pygame.event.get():
        mods = pygame.key.get_mods()
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            match event.key:
                case pygame.K_ESCAPE:
                    running = False 
                case pygame.K_s if (mods & pygame.KMOD_META):
                    to_video() 
                case pygame.K_s if (mods & pygame.KMOD_CTRL):
                    to_video() 

    # # Reset motor speeds at the start of each frame -------------------------------
    # v_l, v_r = 0, 0

    # # Handle key presses for movement and rotation
    # # TODO: should of course only be two actions: v_l and v_r for differential drive 
    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_w] and keys[pygame.K_LEFT]:
    #     # Move forward while turning left
    #     robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, MOVE_SPEED)
    #     robot_angle = (robot_angle - ROTATE_SPEED) % 360  # Adjust angle for turning
    #     v_l, v_r = MOVE_SPEED - 1, MOVE_SPEED  # Left motor slower than right motor

    # elif keys[pygame.K_w] and keys[pygame.K_RIGHT]:
    #     # Move forward while turning right
    #     robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, MOVE_SPEED)
    #     robot_angle = (robot_angle + ROTATE_SPEED) % 360  # Adjust angle for turning
    #     v_l, v_r = MOVE_SPEED, MOVE_SPEED - 1  # Right motor slower than left motor

    # elif keys[pygame.K_w]:
    #     # Move forward
    #     robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, MOVE_SPEED)
    #     v_l, v_r = MOVE_SPEED, MOVE_SPEED  # Both motors move forward at the same speed

    # elif keys[pygame.K_LEFT]:
    #     # Rotate in place counterclockwise
    #     robot_angle = (robot_angle - ROTATE_SPEED) % 360
    #     v_l, v_r = -ROTATE_SPEED, ROTATE_SPEED  # Left motor backward, right motor forward

    # elif keys[pygame.K_RIGHT]:
    #     # Rotate in place clockwise
    #     robot_angle = (robot_angle + ROTATE_SPEED) % 360
    #     v_l, v_r = ROTATE_SPEED, -ROTATE_SPEED  # Right motor backward, left motor forward

    # else:
    #     # No movement
    #     v_l, v_r = 0, 0 
    
    # Reset motor speeds at the start of each frame
    v_l, v_r = 0, 0

    # Handle key presses for motor control
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:  # "W" key controls the left motor
        v_l = MOVE_SPEED
    if keys[pygame.K_UP]:  # Up arrow key controls the right motor
        v_r = MOVE_SPEED

    # Calculate the robot's movement based on motor speeds
    linear_speed = (v_l + v_r) / 2  # Average speed of both motors
    angular_speed = (v_r - v_l) / ROBOT_RADIUS  # Differential rotation

    # Update the robot's position and angle
    robot_angle = (robot_angle + math.degrees(angular_speed)) % 360
    robot_x, robot_y = move_robot(robot_x, robot_y, robot_angle, linear_speed)

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
    save_frame()
    clock.tick(60)

pygame.quit()
