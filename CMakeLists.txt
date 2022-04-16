cmake_minimum_required(VERSION 3.0.0)

project(ort_tensor_api)

set(ONNXRUNTIME_HOME "" CACHE STRING "The path of onnxruntime.")
if(ONNXRUNTIME_HOME STREQUAL "")
    message(FATAL_ERROR "not defined ONNXRUNTIME_HOME")
else()
    message(STATUS "The path of onnxruntime is ${ONNXRUNTIME_HOME}.")
endif()

set(OPENCV_HOME "" CACHE STRING "The path of opencv.")
if(OPENCV_HOME STREQUAL "")
    message(FATAL_ERROR "not defined OPENCV_HOME")
else()
    message(STATUS "The path of OPENCV_HOME is ${OPENCV_HOME}.")
endif()

aux_source_directory(${PROJECT_SOURCE_DIR}/src/ DIR_SRCS)

add_library(ort_tensor_api STATIC ${DIR_SRCS})
add_library(ort_tensor_api::library ALIAS ort_tensor_api)

target_include_directories(ort_tensor_api PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(ort_tensor_api PUBLIC ${ONNXRUNTIME_HOME}/include)
target_include_directories(ort_tensor_api PRIVATE ${OPENCV_HOME}/include)

target_link_directories(ort_tensor_api PUBLIC ${ONNXRUNTIME_HOME}/lib)
target_link_directories(ort_tensor_api PRIVATE ${OPENCV_HOME}/lib)

target_link_libraries(ort_tensor_api PRIVATE onnxruntime)
target_link_libraries(ort_tensor_api PRIVATE opencv_core310 opencv_imgproc310)





