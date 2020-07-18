cmake_minimum_required(VERSION 2.8.12)
project(rsearch)

set(CMAKE_CXX_FLAGS "-Ofast -fPIC -funroll-loops -ffast-math -std=c++11 -fopenmp -Wall -mavx2 -masm=intel -march=native -m64")

execute_process(COMMAND bash ${CMAKE_SOURCE_DIR}/env.sh)

set(RSEARCH_PYTHON ${CMAKE_SOURCE_DIR}/python)
set(RSEARCH_DIR /home/$ENV{USER}/rsearch)
set(RSEARCH_DATA_DIR ${RSEARCH_DIR}/data)

configure_file(${CMAKE_SOURCE_DIR}/configure.h.in ${CMAKE_SOURCE_DIR}/include/configure.h @ONLY)

option(BUILD_DEBUG "option for debug mode" off)

option(FAISS "option for faiss" on)
if (FAISS)
    ADD_DEFINITIONS(-DWITH_FAISS)
endif()

option(GTEST "option for gtest" on)

option(CUDA "option for cuda" off)
if (CUDA)
    ADD_DEFINITIONS(-DWITH_CUDA)
endif()

if (BUILD_DEBUG)
    ADD_DEFINITIONS(-DDEBUG_MODE)
else()
    set(CMAKE_BUILD_TYPE release)
endif()

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/../rsearch-release")
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH} "${CMAKE_INSTALL_PREFIX}")
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# ---------- Add Faiss ----------
if (FAISS)
    set(FAISS_DIR "${CMAKE_SOURCE_DIR}/thirdparty/faiss")
    set(FAISS_INCLUDE "${FAISS_DIR}/include")
    set(FAISS_LIB "${FAISS_DIR}/lib/libfaiss.so")
    include_directories(${FAISS_INCLUDE})
    set(LINK_LIBS ${FAISS_LIB})
endif() 

if (CUDA)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0/)
    find_package(CUDA REQUIRED)
    include(FindCUDA)
endif()

# ---------- Add gtest ----------
set(GTEST_DIR "${CMAKE_SOURCE_DIR}/thirdparty/gtest-1.8.1")
set(GTEST_INCLUDE_DIR "${GTEST_DIR}/include")
set(GTEST_LIBS "${GTEST_DIR}/lib/libgtest.a")
set(GTEST_MAIN_LIBS "${GTEST_DIR}/lib/libgtest_main.a")
include_directories(${GTEST_INCLUDE_DIR})

# ---------- Main ----------

file(GLOB_RECURSE SRC src/*.cpp src/*.cu)
message(${SRC})

add_library(rsearch SHARED ${SRC})
add_library(rsearchStatic STATIC ${SRC})

target_link_libraries(rsearch ${LINK_LIBS})
target_link_libraries(rsearchStatic ${LINK_LIBS})

add_executable(demo test/demo.cpp)
target_link_libraries(demo rsearch)

add_executable(benchmark_rapid_mul test/benchmark/benchmark_rapid_mul)
target_link_libraries(benchmark_rapid_mul rsearch)

add_executable(benchmark_rapid_la test/benchmark/benchmark_rapid_la)
target_link_libraries(benchmark_rapid_la rsearch)

file(GLOB_RECURSE UNIT_TEST_SRC test/unit_test/*.c*)
add_executable(unit_test ${UNIT_TEST_SRC})
target_link_libraries(unit_test rsearch ${GTEST_LIBS} ${GTEST_MAIN_LIBS} )

#file(GLOB_RECURSE UNIT_TEST_SRC test/unit_test/*.c*)
#cuda_add_executable(unit_test ${UNIT_TEST_SRC})



#set(UTILS_HEADER utils/feature_factory.h )
#set(TEST_SRC test/test.cpp )
set(HEADERS include/rsearch.h include/rsearch_type.h)
set(UTILS_HEADER include/utils/utils.h)
install(FILES ${HEADERS} DESTINATION "include")
install(FILES ${UTILS_HEADER} DESTINATION "utils")
install(TARGETS rsearch DESTINATION "lib")
install(TARGETS rsearchStatic DESTINATION "lib")

