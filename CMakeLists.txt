cmake_minimum_required(VERSION 3.28)
project(DCE_quasilinear)

set(CMAKE_CXX_STANDARD 17)

add_executable(run_evolution run_evolution.cpp evolution/evolution.cpp)



set_source_files_properties(
        evolution/evolution.cpp
        PROPERTIES

        COMPILE_OPTIONS "-O3"
)

set_source_files_properties(
        run_evolution.cpp
        PROPERTIES

        COMPILE_OPTIONS "-O3"
)


set(Python3_INCLUDE_DIRS "~/anaconda3/bin/python")
set(Python3_LIBRARIES "~/anaconda3/lib/libpython3.12.so")
include_directories(${Python3_INCLUDE_DIRS})

set(Boost_NO_SYSTEM_PATHS YES)
if (Boost_NO_SYSTEM_PATHS)
    set(Boost_ROOT "~/cppLocal/boostBuild")
    set(Boost_INCLUDE_DIR "~/cppLocal/boostBuild/include")
    set(Boost_LIBRARY_DIR "~/cppLocal/boostBuild/lib")
endif (Boost_NO_SYSTEM_PATHS)

find_package(Boost REQUIRED COMPONENTS python numpy filesystem)

if(Boost_FOUND)
    message(STATUS "boost root is: ${Boost_ROOT}")
    message(STATUS "boost include path is : ${Boost_INCLUDE_DIR}")
    message(STATUS "boost library path is : ${Boost_LIBRARY_DIR}")
    message(STATUS "boost libraries is : ${Boost_LIBRARIES}")
    include_directories(${Boost_INCLUDE_DIR})
    link_directories(${Boost_LIBRARY_DIR})
else()
    message(WARNING "boost not found.")
endif()


target_link_libraries(run_evolution Boost::filesystem)

target_link_libraries(run_evolution Boost::python)
target_link_libraries(run_evolution Boost::numpy)
#python
target_link_libraries(run_evolution ${Python3_LIBRARIES})

include_directories("~/cppLocal/fftw3/include")
include_directories("~/cppLocal/armadillo/include")
#armadillo
file(GLOB LIBS_SO "~/cppLocal/armadillo/lib/libarmadillo.so")
target_link_libraries(run_evolution ${LIBS_SO})


#fftw3
file(GLOB LIBS_A "~/cppLocal/fftw3/lib/*.a")
target_link_libraries(run_evolution ${LIBS_A})