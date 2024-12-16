cmake_minimum_required(VERSION 3.8)
project(bhuman)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

# source files
file(GLOB SOURCE_FILES src/*.cpp src/Math/*.cpp src/MathBase/*.cpp src/Platform/*.cpp src/Streaming/*.cpp)
add_library(bhuman ${SOURCE_FILES})

ament_target_dependencies(bhuman PUBLIC EIGEN3)

target_include_directories(bhuman
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
# ament_export_dependencies(some_dependency)

ament_package()
