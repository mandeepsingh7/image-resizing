# Image Resizing

- This application performs image resizing using custom implementation of interpolation methods (Nearest Neighbor, Bilinear, and Bicubic) and compares them with OpenCV's built-in implementation of those methods.
- It measures and displays the time taken for each implementation.
- It checks for consistency between the custom and built-in implementation.

## Dependencies

- OpenCV (version 4.0.0 or later)
- C++17 or later
- CMake (version 3.10 or later)

## Steps

1. **Install OpenCV:**
      ```bash
   sudo apt-get update
   sudo apt-get install libopencv-dev

2. **Install CMake:**
      ```bash
   sudo apt-get install cmake

3. **Install build-essential:**
      ```bash
   sudo apt-get install -y build-essential

4. **Clone the repository:** 
      ```bash
    git clone https://github.com/mandeepsingh7/image-resizing.git
    cd image-resizing

5. **Build the application:**
      ```bash
    cmake .
    make

6. **Run the application:**
      ```bash
    ./image_resizing

## Notes
- Ensure that the input image file (G178_2 -1080.BMP) is in the same directory as the executable or provide the correct path in the source code.
- Modify the iteration_count in the source code to change the number of iterations for timing tests.
