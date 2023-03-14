cmake_minimum_required(VERSION 3.8.0)

project(ceres_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)

include_directories(
  ${CERES_INCLUDE_DIRS}
)

add_executable(ceres_example
demo.cpp)

target_link_libraries(ceres_example
    ${CERES_LIBRARIES}
)
