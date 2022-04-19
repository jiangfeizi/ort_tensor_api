cmake_minimum_required(VERSION 3.0.0)

project(ort_tensor_api_and_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(DEBUG "create info of debug." ON)
if(DEBUG)
    set (CMAKE_CXX_FLAGS_RELEASE "/MD /Zi /Od /Ob2 /DNDEBUG" CACHE STRING "Set C++ Compiler Flags" FORCE)
    set (CMAKE_SHARED_LINKER_FLAGS_RELEASE "/debug /INCREMENTAL:NO" CACHE STRING "Set C++ Compiler Flags" FORCE)
    set (CMAKE_EXE_LINKER_FLAGS_RELEASE "/debug /INCREMENTAL:NO" CACHE STRING "Set C++ Compiler Flags" FORCE)
else()
    set (CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DNDEBUG" CACHE STRING "Set C++ Compiler Flags" FORCE)
    set (CMAKE_SHARED_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO" CACHE STRING "Set C++ Compiler Flags" FORCE)
    set (CMAKE_EXE_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO" CACHE STRING "Set C++ Compiler Flags" FORCE)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/ort_tensor_api ${PROJECT_SOURCE_DIR}/ort_tensor_api/build) 
add_subdirectory(${PROJECT_SOURCE_DIR}/test ${PROJECT_SOURCE_DIR}/test/build) 


