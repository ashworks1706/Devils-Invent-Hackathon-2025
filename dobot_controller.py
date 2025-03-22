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
    
    def move_to(self, x: float, y: float, z: float, r: float = 0, wait: bool = True):
        """Move the robot to a specific position.
        
        Args:
            x: X coordinate (mm)
            y: Y coordinate (mm)
            z: Z coordinate (mm)
            r: Rotation angle (degrees)
            wait: Whether to wait for movement to complete
        """
        self.bot.set_point_to_point_command(mode=0, x=x, y=y, z=z, r=r)
        if wait:
            sleep(1)  # Basic wait, could be improved with position checking
    
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