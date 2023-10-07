find_package(OpenMP REQUIRED)
message("OpenMP_C_FLAGS = ${OpenMP_C_FLAGS}")
message("OpenMP_CXX_FLAGS = ${OpenMP_CXX_FLAGS}")
message("OpenMP_EXE_LINKER_FLAGS = ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#

set(OpenCV_DIR /usr/local/opencv)
find_package(OpenCV REQUIRED)
message("OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS = ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

add_executable(main main.c)
add_executable(cv_keypoint cv_keypoint.cpp)

file(GLOB PICS ${CMAKE_CURRENT_SOURCE_DIR}/*.jpg)
file(COPY ${PICS} DESTINATION .)
