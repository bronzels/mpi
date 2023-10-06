find_package(MPI REQUIRED)
message("MPI_INCLUDE_PATH = ${MPI_INCLUDE_PATH}")
message("MPI_LIBRARIES = ${MPI_LIBRARIES}")

find_package(OpenMP REQUIRED)
message("OpenMP_C_FLAGS = ${OpenMP_C_FLAGS}")
message("OpenMP_CXX_FLAGS = ${OpenMP_CXX_FLAGS}")
message("OpenMP_EXE_LINKER_FLAGS = ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(COMMON_DIR /data0/shouxiecuda/common)
set(COMMON_CPP ${COMMON_DIR}/common.cpp)

include_directories(${MPI_INCLUDE_PATH} ${COMMON_DIR})
link_libraries(${MPI_LIBRARIES})
#set(CMAKE_CXX_COMPILER mpicxx)
#set(CMAKE_C_COMPILER  mpicc)
#

add_executable(matmul ../mpi_openmp/matmul.cpp ${COMMON_CPP})
add_executable(splitab_orig ../mpi_openmp/splitab_orig.cpp ${COMMON_CPP})

#set(EXECUTABLE_OUTPUT_PATH, "/data0/nfs/mpi")
add_custom_command(
        TARGET matmul POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy matmul /data0/nfs/mpi
)
add_custom_command(
        TARGET splitab_orig POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy splitab_orig /data0/nfs/mpi
)
