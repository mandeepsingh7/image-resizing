cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Image_Resizing)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the source file
add_executable(image_resizing resize_image.cpp)

# Link OpenCV libraries
target_link_libraries(image_resizing ${OpenCV_LIBS})
