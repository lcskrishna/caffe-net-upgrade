
cmake_minimum_required (VERSION 2.8)
project(caffe-net-upgrade)

set (CMAKE_CXX_STANDARD 11)

find_package(Protobuf REQUIRED)
file(DOWNLOAD "https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto" "${CMAKE_CURRENT_BINARY_DIR}/caffe.proto")
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/caffe.proto)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

list(APPEND SOURCES
	src/upgrade_caffe_layers.cpp
        ${PROTO_SRCS} ${PROTO_HDRS}
    )

add_executable(upgrade_caffe_layers ${SOURCES})
target_link_libraries(upgrade_caffe_layers ${PROTOBUF_LIBRARIES})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

