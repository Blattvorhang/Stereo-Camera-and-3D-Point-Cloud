cmake_minimum_required( VERSION 3.5.1 )

project(stereo_camera)

# Find libraries
find_package(OpenCV 4.9.0 REQUIRED)

find_package(PCL REQUIRED COMPONENTS common io visualization filters)
message(STATUS "PCL library status:")
message(STATUS "    config: ${PCL_DIR}")
message(STATUS "    version: ${PCL_VERSION}")
message(STATUS "    libraries: ${PCL_LIBRARY_DIRS}")
message(STATUS "    include path: ${PCL_INCLUDE_DIRS}")

include_directories(/usr/local/include/eigen3)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# set the build type to Release
# set(CMAKE_BUILD_TYPE "Release")

# add the executable
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/stereo_system.cpp
    src/camera.cpp
    src/disparity.cpp
    src/semi_global_matching.cpp
    src/sgm_util.cpp
)

# add the include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${sophus_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${compiler_options})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 17)
