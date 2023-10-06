cmake_minimum_required(VERSION 3.23)
project(mpi)

set(CMAKE_CXX_STANDARD 11)

ADD_SUBDIRECTORY(./mpi)
ADD_SUBDIRECTORY(./openmp)
ADD_SUBDIRECTORY(./mpi_openmp)
ADD_SUBDIRECTORY(./heterogeneous)
