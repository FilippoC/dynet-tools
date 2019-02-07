cmake_minimum_required(VERSION 3.12)
project(dynet-tools)

set(CMAKE_CXX_STANDARD 11)

include_directories("/Users/filippo/repos/dynet")

add_subdirectory(libdytools)

add_executable(dep-parser-train app/src/dep-parser-train.cpp)
target_link_libraries(dep-parser-train libdytools)
target_link_libraries(dep-parser-train dynet)

add_executable(dep-parser-predict app/src/dep-parser-predict.cpp)
target_link_libraries(dep-parser-predict libdytools)
target_link_libraries(dep-parser-predict dynet)
