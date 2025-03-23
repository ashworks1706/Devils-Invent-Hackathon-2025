from dobot_python.lib.interface import Interface
from time import sleep
from typing import List, Optional, Tuple, Union

class MovementException(Exception):
    """Exception raised when movement operations fail."""
    pass

class DobotController:
    def __init__(self, port: Optional[str] = None):
        """Initialize the Dobot controller.
        
        Args:
            port: Serial port to connect to. Defaults to '/dev/ttyACM0' if None.
        """
        if port is None:
            port = '/dev/ttyACM0'
        self.bot = Interface(port)
        if not self.bot.connected():
            raise ConnectionError("Failed to connect to Dobot robot")
        
        # Set default movement parameters
        self.set_movement_speed(velocity=150, acceleration=140)
        self.segment_size = 10  # Size of each movement segment in mm
        self.interpolation_points = 5  # Number of points to interpolate between segments
        
        # Initialize continuous trajectory parameters for smoother movements
        self.bot.set_continous_trajectory_params(150, 150, 150)
        
    def set_movement_speed(self, velocity: float = 100, acceleration: float = 100):
        """Set the movement speed and acceleration for point-to-point movements.
        
        Args:
            velocity: Movement velocity (0-100)
            acceleration: Movement acceleration (0-100)
        """
        self.bot.set_point_to_point_coordinate_params(
            coordinate_velocity=velocity,
            effector_velocity=velocity,
            coordinate_acceleration=acceleration,
            effector_acceleration=acceleration
        )
        # Also update continuous trajectory parameters
        self.bot.set_continous_trajectory_params(acceleration, velocity, acceleration)
    
    def _generate_trajectory(self, start_pos: Tuple[float, float, float, float], 
                           end_pos: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        """Generate a smooth trajectory between two points.
        
        Args:
            start_pos: Starting position (x, y, z, r)
            end_pos: Ending position (x, y, z, r)
            
        Returns:
            List of interpolated positions
        """
        trajectory = []
        for i in range(self.interpolation_points):
            t = i / (self.interpolation_points - 1)
            # Use cubic interpolation for smoother motion
            t = t * t * (3 - 2 * t)  # Smoothstep function
            
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * t
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * t
            z = start_pos[2] + (end_pos[2] - start_pos[2]) * t
            r = start_pos[3] + (end_pos[3] - start_pos[3]) * t
            
            trajectory.append((x, y, z, r))
        return trajectory
    
    def _move_segment(self, start_pos: Tuple[float, float, float, float], 
                     end_pos: Tuple[float, float, float, float], wait: bool = True):
        """Move along a single segment with interpolation.
        
        Args:
            start_pos: Starting position
            end_pos: Ending position
            wait: Whether to wait for movement to complete
        """
        trajectory = self._generate_trajectory(start_pos, end_pos)
        for pos in trajectory:
            self.bot.set_point_to_point_command(mode=2, x=pos[0], y=pos[1], z=pos[2], r=pos[3])
            if wait:
                sleep(0.1)  # Small delay between interpolated points
    
    def move_to(self, x: float, y: float, z: float, r: float = 0, wait: bool = True):
        """Move the robot to a specific position using arc movement for smooth motion.
        
        Args:
            x: X coordinate (mm)
            y: Y coordinate (mm)
            z: Z coordinate (mm)
            r: Rotation angle (degrees)
            wait: Whether to wait for movement to complete
        """
        current_pos = self.get_position()
        target_pos = [x, y, z, r]
        
        # Calculate an intermediate point for the arc
        # This creates a smooth curve between start and end points
        mid_x = (current_pos[0] + x) / 2
        mid_y = (current_pos[1] + y) / 2
        mid_z = (current_pos[2] + z) / 2
        mid_r = (current_pos[3] + r) / 2
        
        # Add a slight offset to create a curve
        offset = 20  # mm
        if abs(x - current_pos[0]) > abs(y - current_pos[1]):
            mid_y += offset
        else:
            mid_x += offset
            
        intermediate_point = [mid_x, mid_y, mid_z, mid_r]
        
        # Execute arc movement through intermediate point
        queue_index = self.bot.set_arc_command(intermediate_point, target_pos)
        
        if wait:
<<<<<<< Updated upstream
            self._wait_for_queue(queue_index)  # Wait for movement to complete using queue index
=======
            sleep(3)  # Wait for movement to complete
>>>>>>> Stashed changes
    
    def move_relative(self, dx: float, dy: float, dz: float, dr: float = 0, wait: bool = True):
        """Move the robot relative to its current position using continuous trajectory for smoother motion.
        
        Args:
            dx: Change in X coordinate (mm)
            dy: Change in Y coordinate (mm)
            dz: Change in Z coordinate (mm)
            dr: Change in rotation angle (degrees)
            wait: Whether to wait for movement to complete
        """
        current_pos = self.get_position()
        target_x = current_pos[0] + dx
        target_y = current_pos[1] + dy
        target_z = current_pos[2] + dz
        target_r = current_pos[3] + dr
        
        # For very small movements, use direct point-to-point
        if abs(dx) < 5 and abs(dy) < 5 and abs(dz) < 5:
            self.bot.set_point_to_point_command(mode=2, x=target_x, y=target_y, z=target_z, r=target_r)
            if wait:
                sleep(0.5)
            return
            
        # For larger movements, use continuous trajectory for smoothness
        # Stop the queue first
        self.bot.stop_queue()
        self.bot.clear_queue()
        
        # Generate a path with more intermediate points for smoother motion
        path_points = 10
        path = []
        
        for i in range(path_points + 1):
            t = i / path_points
            # Apply easing function for acceleration/deceleration
            if t < 0.5:
                # Accelerate (ease in)
                ease = 2 * t * t
            else:
                # Decelerate (ease out)
                ease = 1 - pow(-2 * t + 2, 2) / 2
                
            x = current_pos[0] + dx * ease
            y = current_pos[1] + dy * ease
            z = current_pos[2] + dz * ease
            
            path.append([x, y, z])
        
        # Execute the smooth path
        queue_index = None
        for point in path:
            queue_index = self.bot.set_continous_trajectory_command(1, point[0], point[1], point[2], 50)
            
        # Start queue execution
        self.bot.start_queue()
        
        if wait:
            # Wait for movement to complete using a queue index check
            self._wait_for_queue(queue_index)
            
        # Set rotation separately if needed
        if abs(dr) > 0:
            self.bot.set_point_to_point_command(mode=2, x=target_x, y=target_y, z=target_z, r=target_r)
            if wait:
                sleep(0.5)
    
    def _wait_for_queue(self, queue_index=None):
        """Wait for the queue to complete execution.
        
        Args:
            queue_index: The queue index to wait for completion
        """
        # Add a zero wait command as a non-operation to bypass queue limitations
        self.bot.wait(0)
        
        if queue_index is None:
            queue_index = self.bot.get_current_queue_index()
            
        while True:
            current_index = self.bot.get_current_queue_index()
            if current_index > queue_index:
                break
            sleep(0.2)  # Check more frequently for more responsive behavior
    
    def get_position(self) -> Tuple[float, float, float, float]:
        """Get the current position of the robot.
        
        Returns:
            Tuple of (x, y, z, r) coordinates
        """
        return self.bot.get_pose()
    
    def home(self):
        """Move the robot to its home position."""
        self.bot.set_homing_command(1)
        sleep(2)  # Wait for homing to complete
    
    def set_gripper(self, enable: bool, grip: bool = False):
        """Control the gripper end effector.
        
        Args:
            enable: Whether to enable gripper control
            grip: Whether to grip (True) or release (False)
        """
        self.bot.set_end_effector_gripper(enable_control=enable, enable_grip=grip)
    
    def set_suction_cup(self, enable: bool, suction: bool = False):
        """Control the suction cup end effector.
        
        Args:
            enable: Whether to enable suction cup control
            suction: Whether to activate suction (True) or release (False)
        """
        self.bot.set_end_effector_suction_cup(enable_control=enable, enable_suction=suction)
    
    def set_laser(self, enable: bool, power: int = 0):
        """Control the laser end effector.
        
        Args:
            enable: Whether to enable laser control
            power: Laser power (0-255)
        """
        self.bot.set_end_effector_laser(enable_control=enable, enable_laser=power > 0)
    
    def clear_alarms(self):
        """Clear any active alarms."""
        self.bot.clear_alarms_state()
    
    def get_alarms(self) -> List[int]:
        """Get the current alarm states.
        
        Returns:
            List of active alarm codes
        """
        return self.bot.get_alarms_state()
    
    def wait(self, seconds: float):
        """Wait for a specified number of seconds.
        
        Args:
            seconds: Time to wait in seconds
        """
        sleep(seconds) 
