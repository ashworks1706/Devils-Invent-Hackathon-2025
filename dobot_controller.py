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
        self.set_movement_speed(velocity=100, acceleration=100)
        self.segment_size = 10  # Size of each movement segment in mm
        self.interpolation_points = 5  # Number of points to interpolate between segments
        
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
            self.bot.set_point_to_point_command(mode=0, x=pos[0], y=pos[1], z=pos[2], r=pos[3])
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
        self.bot.set_arc_command(intermediate_point, target_pos)
        
        if wait:
            sleep(1)  # Wait for movement to complete
    
    def move_relative(self, dx: float, dy: float, dz: float, dr: float = 0, wait: bool = True):
        """Move the robot relative to its current position.
        
        Args:
            dx: Change in X coordinate (mm)
            dy: Change in Y coordinate (mm)
            dz: Change in Z coordinate (mm)
            dr: Change in rotation angle (degrees)
            wait: Whether to wait for movement to complete
        """
        current_pos = self.get_position()
        self.move_to(
            x=current_pos[0] + dx,
            y=current_pos[1] + dy,
            z=current_pos[2] + dz,
            r=current_pos[3] + dr,
            wait=wait
        )
    
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