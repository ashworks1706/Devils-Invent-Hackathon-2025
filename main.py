from dobot_controller import DobotController
# Initialize the robot
robot = DobotController()

# Move to a position
robot.move_to(x=200, y=0, z=100)

# Pick up an object with the gripper
robot.set_gripper(enable=True, grip=True)
robot.move_to(x=200, y=0, z=50)
robot.wait(1)
robot.set_gripper(enable=True, grip=True)
robot.move_to(x=200, y=0, z=100)

# Move relative to current position
robot.move_relative(dx=50, dy=0, dz=0)

# Return home
robot.home()