cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CustomImageResizing)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the source file
add_executable(custom_image_resizing resize.cpp)

# Link OpenCV libraries
target_link_libraries(custom_image_resizing ${OpenCV_LIBS})