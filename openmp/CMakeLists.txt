find_package(OpenMP REQUIRED)
message("OpenMP_C_FLAGS = ${OpenMP_C_FLAGS}")
message("OpenMP_CXX_FLAGS = ${OpenMP_CXX_FLAGS}")
message("OpenMP_EXE_LINKER_FLAGS = ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#

add_executable(main main.c)
