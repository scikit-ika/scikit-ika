cmake_minimum_required(VERSION 3.10)

project(pearl VERSION 0.0.1)

set(sourcefiles
skika/pearl/pearl.cpp
skika/pearl/lru_state.cpp
skika/pearl/lossy_state_graph.cpp
)

set(pybindfiles
skika/pearl/pearl_bindings.cpp
)

set(include_dirs
third_party/streamDM/
)

set(glog "-lglog -lpthread")
set(warning "-Wall -Wextra -Wno-sign-compare -Wno-nonnull -Wno-unused-variable")
set(flags "-fPIC -std=c++14 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O3 -DUNIX -g")

add_definitions(${flags})

add_subdirectory(third_party/streamDM)

if(ADD_PEARL_AS_LIBRARY)
	message("PEARL: create library")
	add_library(pearl STATIC ${sourcefiles})
else()
	message("PEARL: turn on Python Wrapper")
	add_subdirectory(third_party/pybind11)
	pybind11_add_module(pearl SHARED ${sourcefiles} ${pybindfiles})
endif()

target_link_libraries(pearl PUBLIC streamDM)
target_include_directories(pearl PUBLIC ${include_dirs})
