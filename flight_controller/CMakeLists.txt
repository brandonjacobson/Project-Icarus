cmake_minimum_required(VERSION 3.16)
project(icarus_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED)

add_library(plant src/plant.cpp)
target_include_directories(plant PUBLIC include)
target_link_libraries(plant PUBLIC Eigen3::Eigen)

add_executable(test_plant tests/test_plant.cpp)
target_link_libraries(test_plant PRIVATE plant)