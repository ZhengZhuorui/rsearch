cmake_minimum_required(VERSION 2.8)
project(rsearch)

set(CMAKE_CXX_FLAGS "-fPIC -std=c++11 -fopenmp -Wall -mavx2 -O3")

option(BUILD_DEBUG "option for debug mode" off)

option(FAISS "option for faiss" off)
if (FAISS)
    ADD_DEFINITIONS(-DWITH_FAISS)
endif()

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
    set(FAISS_DIR "${CMAKE_SOURCE_DIR}/thirdparty/faiss-1.5")
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


file(GLOB_RECURSE SRC src/*.cpp src/*.cu)

if (CUDA)
    cuda_add_library(rsearch SHARED ${SRC})
    cuda_add_library(rsearch STATIC ${SRC})
    
else()
    add_library(rsearch SHARED ${SRC})
    add_library(rsearchStatic STATIC ${SRC})
endif()

target_link_libraries(rsearch ${LINK_LIBS})
target_link_libraries(rsearchStatic ${LINK_LIBS})

#set_target_properties(Rapid1vN PROPERTIES VERSION ${RAPID1vN_VERSION_STRING} SOVERSION ${RAPID1vN_VERSION_MAJOR}.${RAPID1vN_VERSION_MINOR})

add_executable(demo test/demo.cpp)
target_link_libraries(demo rsearch)


#file(GLOB_RECURSE UNIT_TEST_SRC test/unit_test/*.c*)
#cuda_add_executable(unit_test ${UNIT_TEST_SRC})
#target_link_libraries(unit_test Rapid1vN ${GTEST_LIBS} ${GTEST_MAIN_LIBS})

set(HEADERS include/rapid1vn.h include/rapid1vn_gallery.h include/rapid1vn_types.h)
set(UTILS_HEADER utils/feature_factory.h )
set(TEST_SRC test/test.cpp )

install(FILES ${HEADERS} DESTINATION "include")
install(FILES ${TEST_SRC} DESTINATION "test")
install(TARGETS rsearch DESTINATION "lib")
install(TARGETS rsearchStatic DESTINATION "lib")
