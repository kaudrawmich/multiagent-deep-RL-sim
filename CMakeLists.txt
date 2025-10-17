cmake_minimum_required(VERSION 3.8)
project(multiagent_deep_rl_sim)

find_package(ament_cmake REQUIRED)

# Install launch files (create launch/ if needed)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  OPTIONAL
)

# Install your assets
install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

install(DIRECTORY bridge/
  DESTINATION share/${PROJECT_NAME}/bridge
)

install(DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS)

install(PROGRAMS
  nodes/box_push_observation_node.py
  nodes/box_push_dqn_policy_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
