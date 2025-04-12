import os 
import pygame
import math 
import numpy as np
from numpy import pi 
import random 
import datetime as dt 
from PIL import Image 
import imageio.v2 as imageio 


# Screen 
SCREEN_SIZE = 800 
CELL_SIZE = SCREEN_SIZE//20
SCREEN_CAPTURE = True 
# Map
OBSTACLE_DENSITY = 0.2 
# Colors
BACKGROUND_COLOR = (30, 30, 30)
ROBOT_COLOR = (200, 50, 50)
SENSOR_COLOR = (50, 200, 50)
OBSTACLE_COLOR = (100, 100, 100)
SENSOR_HIT_COLOR = (230, 0, 0)
# Robot
ROBOT_RADIUS = 10
MOVE_SPEED = 3
ROTATE_SPEED = 0.2
NUM_SENSORS = 12
SENSOR_LENGTH = 60
SENSOR_ANGLE_STEP = 2*pi/NUM_SENSORS
SENSOR_STEP_SIZE = 5
# Visibility flags 
obstacles_visible = True 
robot_info_visible = True 


class Environment:

    # Initialization ---------------------------------------------------------------------------------------------
    
    def __init__(self):
        self.initialize_map()
        self.initialize_robot()
        if SCREEN_CAPTURE:
            self.prev_frame = None 
            self.frames = []
        
    def initialize_map(self):
        """
        Generates a map with randomly placed blocks.
        """
        self.obstacles = []
        center_rect = pygame.Rect(SCREEN_SIZE/2-2*CELL_SIZE/2, SCREEN_SIZE/2-2*CELL_SIZE/2, 2*CELL_SIZE, 2*CELL_SIZE)
        # Calculate the number of obstacles based on density 
        n_obstacles = int((SCREEN_SIZE**2)/(CELL_SIZE**2)*OBSTACLE_DENSITY)
        while len(self.obstacles) < n_obstacles:
            x = random.randint(0, (SCREEN_SIZE//CELL_SIZE)-1)*CELL_SIZE
            y = random.randint(0, (SCREEN_SIZE//CELL_SIZE)-1)*CELL_SIZE
            # Create the obstacle rectangle
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            # Avoid overlapping obstacles and don't place obstacles in center
            if not (rect.colliderect(center_rect) or any(rect.colliderect(obstacle) for obstacle in self.obstacles)):
                self.obstacles.append(rect)
        # Add boundary walls to fully enclose the map
        self.obstacles.extend([
            pygame.Rect(0, 0, SCREEN_SIZE, 5),              # Top boundary
            pygame.Rect(0, 0, 5, SCREEN_SIZE),              # Left boundary
            pygame.Rect(0, SCREEN_SIZE-5, SCREEN_SIZE, 5),  # Bottom boundary
            pygame.Rect(SCREEN_SIZE-5, 0, 5, SCREEN_SIZE)   # Right boundary
        ])

    def initialize_robot(self):
        # Pose 
        # Important: robot_x, robot_y store the robot's absolute position!
        #  The robot would generally only have access to its position relative to the spawn point
        #  i.e. (robot_x-spawn_x, robot_y-spawn_y) 
        self.robot_x = SCREEN_SIZE/2  # x-position
        self.robot_y = SCREEN_SIZE/2  # y-position 
        self.robot_angle = 0          # orientation (in radians)
        # Find a good spawn point 
        #  Make sure the robot is not placed on top of or inside an obstacle: 
        while self.is_colliding():
            # Reposition the robot's spawn randomly within the map bounds:
            self.robot_x = random.randint(ROBOT_RADIUS, SCREEN_SIZE-ROBOT_RADIUS)
            self.robot_y = random.randint(ROBOT_RADIUS, SCREEN_SIZE-ROBOT_RADIUS)
        # Save this point as (spawn_x, spawn_y) 
        self.spawn_x = self.robot_x 
        self.spawn_y = self.robot_y 
        # Sensors
        # Get initial sensor reading 
        self.compute_sensors()
        # Just for visualization, also keep track of these values
        self.v_l = 0
        self.v_r = 0

    # Step: transition to the next state -------------------------------------------------------------------------
    
    def step(self):
        """
        Transition from current environment state to the next 
        """
        self.compute_sensors()
        if SCREEN_CAPTURE:
            self.save_frame()
        self.render()
        clock.tick(60) 
    
    # Motion -----------------------------------------------------------------------------------------------------

    def move(self, action):
        """
        Parameter 'action' must be a tuple containing (v_l, v_r)
        Move robot at velocities v_l (left-motor velocity), v_r (right-motor velocity) 
        """
        v_l, v_r = action
        self.v_l, self.v_r = v_l, v_r 
        # When passing v_l=1, this is already in units of MOVE_SPEED, so to get absolute units
        # multiply by MOVE_SPEED: 
        v_l *= MOVE_SPEED
        v_r *= MOVE_SPEED 
        # Calculate linear and angular velocities from v_l, v_r 
        v_linear = (v_l + v_r) / 2  # Average speed of both motors
        v_angular = (v_r - v_l) / ROBOT_RADIUS  # Differential rotation
        new_angle = (self.robot_angle + v_angular * ROTATE_SPEED) % (2*pi)
        # Calculate new position based on current angle
        dx = v_linear * math.cos(new_angle)
        dy = v_linear * math.sin(new_angle)
        # Calculate the new position
        new_x = self.robot_x + dx
        new_y = self.robot_y + dy
        # Collision check 
        robot_rect = pygame.Rect(new_x-ROBOT_RADIUS, new_y-ROBOT_RADIUS, ROBOT_RADIUS*2, ROBOT_RADIUS*2)
        x_clear = True
        y_clear = True
        for obs in self.obstacles:
            if robot_rect.colliderect(obs):
                # Check for sliding along the wall
                # Try moving only in the x direction
                temp_x = self.robot_x + dx
                temp_rect_x = pygame.Rect(temp_x-ROBOT_RADIUS, self.robot_y-ROBOT_RADIUS, ROBOT_RADIUS*2, ROBOT_RADIUS*2)
                if temp_rect_x.colliderect(obs):
                    x_clear = False
                # Try moving only in the y direction
                temp_y = self.robot_y + dy
                temp_rect_y = pygame.Rect(self.robot_x-ROBOT_RADIUS, temp_y-ROBOT_RADIUS, ROBOT_RADIUS*2, ROBOT_RADIUS*2)
                if temp_rect_y.colliderect(obs):
                    y_clear = False
        # Resolve movement based on collision checks
        new_angle = (self.robot_angle + v_angular * ROTATE_SPEED) % (2*pi)
        if not x_clear and not y_clear:
            # Block movement completely if both directions are blocked
            # Only change angle 
            self.robot_angle = new_angle 
        elif not x_clear:
            # Allow sliding along the y-axis
            self.robot_y = new_y 
            self.robot_angle = new_angle
        elif not y_clear:
            # Allow sliding along the x-axis
            self.robot_x = new_x 
            self.robot_angle = new_angle
        else:
            # If no collision, move to the new position
            self.robot_x = new_x
            self.robot_y = new_y
            self.robot_angle = new_angle
    
    # Sensors ----------------------------------------------------------------------------------------------------

    def compute_sensors(self):
        sensors = []
        for i in range(NUM_SENSORS): 
            sensor_angle = self.robot_angle + i * SENSOR_ANGLE_STEP
            hit = False 
            distance = math.inf 
            sensor_x, sensor_y = self.robot_x, self.robot_y 
            # Step along the line to detect the closest intersection
            for d in range(0, SENSOR_LENGTH, SENSOR_STEP_SIZE):  
                sensor_x = self.robot_x + d * math.cos(sensor_angle)
                sensor_y = self.robot_y + d * math.sin(sensor_angle)
                point = int(sensor_x), int(sensor_y)
                # Check collision with obstacles
                for obs in self.obstacles:
                    if obs.collidepoint(point):
                        hit = True
                        # Store the distance where collision occurs, in "sensor steps":
                        distance = max((d-ROBOT_RADIUS)//SENSOR_STEP_SIZE, 0) 
                        break
                if hit:
                    break
            # Important: these are ground-truth sensor measurements
            #  Robot only has access to noisy (hit, distance) data 
            sensors.append((sensor_x, sensor_y, hit, distance))
        self.sensors = sensors 

    # Helpers ----------------------------------------------------------------------------------------------------
    
    def is_colliding(self, xy=None):
        """
        Checks if a given position, (x, y), which might be the actual or a hypothetical position
        of the robot, collides with any obstacles
        """
        if xy is None:
            x, y = self.robot_x, self.robot_y 
        else:
            x, y = xy 
        robot_rect = pygame.Rect(x-ROBOT_RADIUS, y-ROBOT_RADIUS, ROBOT_RADIUS*2, ROBOT_RADIUS*2)
        for obs in self.obstacles:
            if robot_rect.colliderect(obs):
                return True
        return False

    def print_sensors(self):
        for i, (sx, sy, hit, distance) in enumerate(self.sensors):
            status = "HIT" if hit else "CLEAR"
            if hit: 
                print(f"Sensor {i+1}: {status} at ({sx:.1f}, {sy:.1f}), distance {distance}")

    # Drawing ----------------------------------------------------------------------------------------------------
    
    def draw_obstacles(self):
        for obs in self.obstacles: 
            pygame.draw.rect(screen, OBSTACLE_COLOR, obs) 
    
    def draw_robot(self):
        global robot_info_visible 
        x, y = self.robot_x, self.robot_y 
        # Draw sensors 
        if robot_info_visible:
            for sensor_x, sensor_y, hit, _ in self.sensors:
                color = SENSOR_HIT_COLOR if hit else SENSOR_COLOR
                pygame.draw.line(screen, color, (x, y), (sensor_x, sensor_y), 2)
        # Draw the robot body (disk)
        pygame.draw.circle(screen, ROBOT_COLOR, (int(x), int(y)), ROBOT_RADIUS)
        # Draw the direction line (heading)
        heading_x = x + ROBOT_RADIUS * math.cos(self.robot_angle)
        heading_y = y + ROBOT_RADIUS * math.sin(self.robot_angle)
        pygame.draw.line(screen, (0, 0, 0), (x, y), (heading_x, heading_y), 3)
        # Draw sensor distance values around the robot (text)
        if robot_info_visible:
            # Draw motor speed text inside the robot body
            # vel_text = font.render(f"[{self.v_l}, {self.v_r}]", True, (255, 255, 255))
            # screen.blit(vel_text, (x-11, y-6))  
            for i, (_, _, hit, distance) in enumerate(self.sensors):
                sensor_angle = self.robot_angle + i * SENSOR_ANGLE_STEP
                text_x = x + (ROBOT_RADIUS + 20) * math.cos(sensor_angle)  # Position outside the robot
                text_y = y + (ROBOT_RADIUS + 20) * math.sin(sensor_angle)  
                # Only draw if finite distance 
                if distance < math.inf:
                    distance_text = font.render(str(int(distance)), True, (255, 255, 255) if hit else (100, 100, 100))
                    text_rect = distance_text.get_rect(center=(text_x, text_y))
                    screen.blit(distance_text, text_rect)
    
    def save_frame(self): 
        frame_surface = pygame.display.get_surface().copy()
        frame_array3d = pygame.surfarray.array3d(frame_surface) 
        # Only add if frame is not empty and different from last frame 
        if (frame_array3d.sum() > 0) and (self.prev_frame is None or (frame_array3d != self.prev_frame).any()):
            frame_array = pygame.image.tostring(frame_surface, "RGB")
            frame = Image.frombytes("RGB", frame_surface.get_size(), frame_array)
            self.frames.append(frame) 
            self.prev_frame = frame_array3d 
    
    def to_video(self):
        if len(self.frames) == 0: 
            return
        if not os.path.exists("screencapture"):
            os.mkdir("screencapture")
        timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"screencapture/main {timestamp}.mp4"
        print(f"Saving file as {filename}") 
        # Save first frame as image
        # with imageio.get_writer(f"screencapture/main {timestamp}.png") as writer: 
        #     writer.append_data(np.array(frames[1]))
        with imageio.get_writer(filename, fps=60, codec="libx264") as writer: 
            for frame in self.frames[1:]:
                writer.append_data(np.array(frame))
    
    def render(self):
        global obstacles_visible 
        screen.fill(BACKGROUND_COLOR)
        if obstacles_visible:
            self.draw_obstacles()
        self.draw_robot()
        pygame.display.flip()


if __name__ == "__main__":
    # PyGame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Mobile Robot")
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont("Calibri", 12)
    
    # Environment 
    env = Environment() 
    
    # Main loop
    running = True 
    while running:
        # env.print_sensors()

        # System controls
        mods = pygame.key.get_mods()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 
            elif event.type == pygame.KEYDOWN:
                match event.key: 
                    case pygame.K_ESCAPE: 
                        running = False 
                    case pygame.K_s if (mods & (pygame.KMOD_META | pygame.KMOD_CTRL)):
                        # CMD + "S" key saves run so far as mp4 file 
                        env.to_video()
                    case pygame.K_r:
                        # "R" key toggles robot info visibility
                        robot_info_visible = not robot_info_visible
                    case pygame.K_o:
                        # "O" key toggles obstacle visibility
                        obstacles_visible = not obstacles_visible
        
        # Robot controls 
        v_l, v_r = 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:  
            # "W" key controls the left motor
            v_l = 1
        if keys[pygame.K_UP]:  
            # UP arrow key controls the right motor
            v_r = 1
        # Move the robot according to (v_l, v_r)
        env.move((v_l, v_r))

        # Transition to next state 
        env.step()
    
    pygame.quit() 
