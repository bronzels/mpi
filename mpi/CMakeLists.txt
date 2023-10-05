find_package(MPI REQUIRED)
message("MPI_INCLUDE_PATH = ${MPI_INCLUDE_PATH}")
message("MPI_LIBRARIES = ${MPI_LIBRARIES}")
include_directories(${MPI_INCLUDE_PATH})
link_libraries(${MPI_LIBRARIES})
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER  mpicc)
#

add_executable(hello hello.c)
add_executable(pi pi.c)
add_executable(matmul matmul.cpp)

#set(EXECUTABLE_OUTPUT_PATH, "/data0/nfs/mpi")
add_custom_command(
        TARGET matmul POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy matmul /data0/nfs/mpi
)
