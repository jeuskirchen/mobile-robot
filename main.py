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
OBSTACLE_DENSITY = 0.2  # 0.2
# Colors
BACKGROUND_COLOR = (30, 30, 30)
ROBOT_COLOR = (200, 50, 50)
SENSOR_COLOR = (50, 200, 50)
OBSTACLE_COLOR = (100, 100, 100)
SENSOR_HIT_COLOR = (230, 0, 0)
# Robot
ROBOT_RADIUS = 10
MOVE_SPEED = 3
ROTATE_SPEED = 0.75
NUM_SENSORS = 12
SENSOR_LENGTH = 60
SENSOR_ANGLE_STEP = 2*pi/NUM_SENSORS
SENSOR_STEP_SIZE = 5
# Visibility flags 
flag_changed = False 
obstacles_visible = True 
robot_info_visible = True 
trajectory_visible = True  
# Kalman filter settings
# BITMAP_SIZE = 100
INITIAL_VAR = np.array([0.1, 0.1, 0.1])
R = np.diag([0.1, 0.1, 0.1]) 


class Environment:

    # Initialization ---------------------------------------------------------------------------------------------
    
    def __init__(self):
        self.timestep = 0 
        self.initialize_map()
        self.initialize_robot()
        self.initialize_belief()
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
        # Important: robot_x, robot_y store the robot's relative position!
        #  To get the absolute position, add the spawn position (spawn_x, spawn_y)
        self.robot_x = 0
        self.robot_y = 0
        self.robot_angle = 0          # orientation (in radians)
        # Find a good spawn point 
        self.spawn_x = SCREEN_SIZE/2  # x-position
        self.spawn_y = SCREEN_SIZE/2  # y-position 
        # Make sure the robot is not placed on top of or inside an obstacle: 
        while self.is_colliding():
            # Reposition the robot's spawn randomly within the map bounds:
            self.spawn_x = random.randint(ROBOT_RADIUS, SCREEN_SIZE-ROBOT_RADIUS)
            self.spawn_y = random.randint(ROBOT_RADIUS, SCREEN_SIZE-ROBOT_RADIUS)
        # List to keep track of robot's trajectory (position in each timestep)
        self.trajectory = [(0, 0)]
        # Sensors
        # Get initial sensor reading 
        self.compute_sensors()
        # Just for visualization, also keep track of these values inside self
        # (rather than just passing them as parameters to the move method)
        self.v_l = 0
        self.v_r = 0
    
    def initialize_belief(self):
        # Belief is the robot's belief of its own pose in absolute terms! 
        # Initialize belief at spawn point (local localization!) 
        #  and set small variance.
        #  For global localization, use random initial point and large variance.
        self.belief_mean = np.array([self.spawn_x, self.spawn_y, self.robot_angle])
        self.belief_var = INITIAL_VAR 
        # Initialize belief trajectory 
        self.belief_trajectory = [self.belief_mean[:2]]

    # Step: transition to the next state -------------------------------------------------------------------------
    
    def step(self, action):
        """
        Transition from current environment state to the next 
        """
        global flag_changed
        self.compute_sensors()
        self.update_belief(action) 
        if SCREEN_CAPTURE:
            self.save_frame()
        self.render()
        self.timestep += 1
        flag_changed = False 
        clock.tick(60) 
    
    # Motion -----------------------------------------------------------------------------------------------------

    def move(self, action):
        """
        Parameter 'action' must be a tuple containing (v_l, v_r)
        Move robot at velocities v_l (left motor), v_r (right motor) 
        """
        if self.timestep > 0 and action == (0, 0) and not flag_changed:
            # If there is no movement, no need to calculate new pose
            #  Unless it's timestep 0; so it renders the map and initial pose
            #  This also means that time doesn't pass, if no action is taken 
            return 
        v_l, v_r = action
        # Keep track of v_l, v_r so we can put it on screen as text 
        #  (see draw_robot method)
        self.v_l, self.v_r = v_l, v_r 
        # Calculate linear and angular velocities from v_l, v_r 
        v_linear = MOVE_SPEED * (v_l + v_r) / 2  # Average speed of both motors
        v_angular = ROTATE_SPEED * (v_r - v_l) / ROBOT_RADIUS  # Differential rotation
        # Calculate new position based on current angle
        d_x = v_linear * math.cos(self.robot_angle)
        d_y = v_linear * math.sin(self.robot_angle)
        d_angle = v_angular
        # Calculate the new position
        new_x = self.robot_x + d_x
        new_y = self.robot_y + d_y
        new_angle = self.robot_angle + d_angle 
        # Make sure angle stays in range [-2π, 2π]:
        new_angle %= 2*pi 
        # TODO: add motion noise (I assume we have to use matrix R here?) 
        #
        #
        # Collision check 
        robot_rect = pygame.Rect(self.spawn_x + new_x - ROBOT_RADIUS, 
                                 self.spawn_y + new_y - ROBOT_RADIUS, 
                                 ROBOT_RADIUS * 2, 
                                 ROBOT_RADIUS * 2)
        x_clear = True
        y_clear = True
        for obs in self.obstacles:
            if robot_rect.colliderect(obs):
                # Check for sliding along the wall
                # Try moving only in the x direction
                temp_x = self.robot_x + d_x
                temp_rect_x = pygame.Rect(self.spawn_x + temp_x - ROBOT_RADIUS, 
                                          self.spawn_y + self.robot_y - ROBOT_RADIUS, 
                                          ROBOT_RADIUS * 2, 
                                          ROBOT_RADIUS * 2)
                if temp_rect_x.colliderect(obs):
                    x_clear = False
                # Try moving only in the y direction
                temp_y = self.robot_y + d_y
                temp_rect_y = pygame.Rect(self.spawn_x + self.robot_x - ROBOT_RADIUS, 
                                          self.spawn_y + temp_y - ROBOT_RADIUS, 
                                          ROBOT_RADIUS * 2, 
                                          ROBOT_RADIUS * 2)
                if temp_rect_y.colliderect(obs):
                    y_clear = False
        # Resolve movement based on collision checks
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
        # Save new position in trajectory
        self.trajectory.append((self.robot_x, self.robot_y))
        # Transition to next state 
        env.step(action) 

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
                # point = int(sensor_x), int(sensor_y)
                # Use absolute position just to check collision 
                point = self.spawn_x + sensor_x, self.spawn_y + sensor_y 
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
            #  Also, we only append relative positions!! 
            #  Otherwise, the robot would know its exact position from the sensor readings!
            sensors.append((sensor_x, sensor_y, hit, distance))
        self.sensors = sensors 

    def compute_observation(self):
        # "observation" = triangulated/trilatered absolute robot position from sensor data 
        observation = None 
        # TODO
        #
        #
        #
        # TODO: add sensor noise (I assume we have to use matrix Q here?)
        # 
        #
        #
        return observation

    # Kalman filter ----------------------------------------------------------------------------------------------

    def update_belief(self, action): 
        """
        Updates the belief using the Kalman filter 
        Parameter 'action' must be a tuple containing (v_l, v_r) 
        """
        # Motion update ("Prediction")
        #  Simple model of how model moves given the action without 
        #  any consideration of obstacles
        v_l, v_r = action 
        v_linear = MOVE_SPEED * (v_l + v_r) / 2  # Average speed of both motors
        v_angular = ROTATE_SPEED * (v_r - v_l) / ROBOT_RADIUS  # Differential rotation
        # Believed orientation:
        angle = self.belief_mean[2]
        # u vector: action (we're using linear velocity, angular velocity here)
        u = np.array([v_linear, v_angular])  
        # A matrix (effect of environment on next state)
        #  Assume environment has no effect -> use identity matrix 
        #  So we can just leave it 
        # B matrix (effect of taking action u on next state)
        B = np.array([
            [np.cos(angle), 0],
            [np.sin(angle), 0],
            [0,             1]
        ])
        # Update mean 
        self.belief_mean = self.belief_mean + B.dot(u) 
        # Make sure angle stays in range [-2π, 2π]:
        self.belief_mean[2] %= 2*pi  
        # Update covariance (belief_var)
        #  Actually just the diagonal of the covariance matrix, as we assume independence, 
        #  i.e. just the variances 
        belief_cov = np.diag(self.belief_var) 
        self.belief_var = (belief_cov + R).diagonal()  # turn diagonal back into vector 
        print(self.belief_var)
        
        # Sensor update ("Correction")
        # We got self.sensors with entries (sx, sy, hit, distance) 
        # using triangulation or trilateration 
        observation = self.compute_observation()
        # TODO
        # TODO: might be easier, if we create an occupancy bitmap first and then use this as the map?
        #  Inside the occupancy bitmap, we could also limit ourselves to a small window around the 
        #  mean believed position according to the motion model, 
        #  and the size of the window could depend on the variance 
        # 
        # 
        # 

        # Append believed pose to belief trajectory
        self.belief_trajectory.append(self.belief_mean[:2])

    # Helpers ----------------------------------------------------------------------------------------------------
    
    def is_colliding(self, xy=None):
        """
        Checks if a given position, (x, y), which might be the actual or a hypothetical position
        of the robot, collides with any obstacles
        """
        if xy is None:
            x, y = self.spawn_x + self.robot_x, self.spawn_y + self.robot_y 
        else:
            x, y = xy 
        robot_rect = pygame.Rect(x-ROBOT_RADIUS, y-ROBOT_RADIUS, ROBOT_RADIUS*2, ROBOT_RADIUS*2)
        for obs in self.obstacles:
            if robot_rect.colliderect(obs):
                return True
        return False

    def print_sensors(self):
        for i, (sx, sy, hit, distance) in enumerate(self.sensors):
            # sx, sy are sensor positions relative to the robot
            # to get absolute ones add spawn_x, spawn_y respectively 
            status = "HIT" if hit else "CLEAR"
            if hit: 
                print(f"Sensor {i+1}: {status} at ({sx:.1f}, {sy:.1f}), distance {distance}")
                # print(f"Sensor {i+1}: {status} at ({sx+spawn_x:.1f}, {sy+spawn_y:.1f}), distance {distance}")

    # Drawing ----------------------------------------------------------------------------------------------------
    
    def draw_obstacles(self):
        if not obstacles_visible:
            return 
        for obs in self.obstacles: 
            pygame.draw.rect(screen, OBSTACLE_COLOR, obs) 
    
    def draw_robot(self):
        # Draw trajectory (past positions)
        if trajectory_visible:
            for x, y in self.trajectory:
                pygame.draw.circle(screen, (90, 30, 30), (self.spawn_x+x, self.spawn_y+y), 3)
        # Get robot's current (absolute) position
        x, y = self.spawn_x + self.robot_x, self.spawn_y + self.robot_y 
        # Draw sensors 
        if robot_info_visible:
            for sensor_x, sensor_y, hit, _ in self.sensors:
                color = SENSOR_HIT_COLOR if hit else SENSOR_COLOR
                pygame.draw.line(screen, color, (x, y), (self.spawn_x+sensor_x, self.spawn_y+sensor_y), 2)
                pygame.draw.circle(screen, color, (self.spawn_x+sensor_x, self.spawn_y+sensor_y), 3)
        # Draw the robot body (disk)
        pygame.draw.circle(screen, ROBOT_COLOR, (x, y), ROBOT_RADIUS)
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

    def draw_belief(self):
        # Draw trajectory (i.e. past believed positions)
        if trajectory_visible:
            for x, y in self.belief_trajectory:
                pygame.draw.circle(screen, (30, 30, 90), (x, y), 3)
        # Get current believed position (mean and variance)
        mean_x, mean_y, mean_angle = self.belief_mean
        '''
        var_x, var_y, var_angle = self.belief_var 
        # Draw covariance
        # pygame.draw.circle(screen, (30, 30, 200), (mean_x, mean_y), (var_x+var_y)/2, 2)
        ellipse = pygame.Rect(mean_x - var_x/2, mean_y - var_y/2, var_x, var_y)
        # pygame.draw.rect(screen, (30, 30, 200), ellipse, 2) 
        pygame.draw.ellipse(screen, (30, 30, 200), ellipse, 2)
        '''
        # Draw the robot body (disk) at (mean) believed location 
        pygame.draw.circle(screen, (50, 50, 255), (mean_x, mean_y), ROBOT_RADIUS)
        # Draw the direction line (heading) of (mean) believed orientation 
        heading_x = mean_x + ROBOT_RADIUS * math.cos(mean_angle)
        heading_y = mean_y + ROBOT_RADIUS * math.sin(mean_angle)
        pygame.draw.line(screen, (0, 0, 0), (mean_x, mean_y), (heading_x, heading_y), 3)

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
        screen.fill(BACKGROUND_COLOR)
        self.draw_obstacles()
        self.draw_belief()
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
                flag_changed = True
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
                    case pygame.K_t:
                        # "T" key toggles trajectory visibility
                        trajectory_visible = not trajectory_visible
        
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
    
    pygame.quit() 
