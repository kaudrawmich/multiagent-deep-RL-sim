import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/julia/ros2_ws/multiagent-deep-RL-sim/install/box_pushing_agent'
