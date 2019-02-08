cmake_minimum_required(VERSION 3.12)
project(dynet-tools)

set(CMAKE_CXX_STANDARD 11)

add_definitions("-Wall")
add_definitions("-Wextra")
add_definitions("-pedantic")
add_definitions("-O3")
add_definitions("-Wfatal-errors")
add_definitions("-g")
add_definitions("-std=c++11")
add_definitions("-funroll-loops")
add_definitions("-DEIGEN_FAST_MATH")
add_definitions("-march=native")
add_definitions("-DBOOST_LOG_DYN_LINK")


include_directories("/Users/filippo/repos/dynet")

add_subdirectory(libdytools)

add_executable(dep-parser-train app/src/dep-parser-train.cpp)
target_link_libraries(dep-parser-train libdytools)
target_link_libraries(dep-parser-train dynet)

add_executable(dep-parser-predict app/src/dep-parser-predict.cpp)
target_link_libraries(dep-parser-predict libdytools)
target_link_libraries(dep-parser-predict dynet)
