cmake_minimum_required(VERSION 3.0.0)

project(test)

aux_source_directory(${PROJECT_SOURCE_DIR}/ TEST_SRCS)

configure_file(config.h.in config.h)

add_executable(test ${TEST_SRCS})

target_include_directories(test PRIVATE ${PROJECT_BINARY_DIR})

target_link_libraries(test PRIVATE ort_tensor_api::library)
target_link_libraries(test PRIVATE opencv_core310 opencv_imgcodecs310 opencv_imgproc310)


