find_package(MPI REQUIRED)
message("MPI_INCLUDE_PATH = ${MPI_INCLUDE_PATH}")
message("MPI_LIBRARIES = ${MPI_LIBRARIES}")

set(COMMON_DIR /data0/shouxiecuda/common)
set(COMMON_CPP ${COMMON_DIR}/common.cpp)

include_directories(${MPI_INCLUDE_PATH} ${COMMON_DIR})
link_libraries(${MPI_LIBRARIES})
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER  mpicc)
#

