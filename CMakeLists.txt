cmake_minimum_required( VERSION 3.5.1 )

project(stereo_camera)

# set the C++ standard to C++ 14
set(CMAKE_CXX_STANDARD 14)

# Find libraries
find_package(OpenCV 4.9.0 REQUIRED)
find_package(Sophus REQUIRED)
find_package(GeographicLib REQUIRED)
find_package(fmt REQUIRED)
set (FMT_LIBRARIES fmt::fmt)
include_directories(/usr/local/include/eigen3)

# set the build type to Release
# set(CMAKE_BUILD_TYPE "Release")

# add the executable
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/stereo_system.cpp
    src/camera.cpp
    src/disparity.cpp
)

# add the include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${GeographicLib_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Sophus_LIBRARIES} fmt)
# target_link_libraries(${PROJECT_NAME} PRIVATE ${fmt_LIBS})
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${sophus_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${compiler_options})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
