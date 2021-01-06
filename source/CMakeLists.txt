cmake_minimum_required(VERSION 3.19)

project(Word2Vec
        VERSION 1.0
        LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -DDEBUG -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

add_executable(word2vec word2vec.cpp)
add_executable(word2vec2 word2vec2.cpp)
add_executable(distance distance.cpp)

target_link_libraries(distance boost_program_options tbb)
target_link_libraries(word2vec boost_program_options tbb pthread)
target_link_libraries(word2vec2 boost_program_options tbb)
