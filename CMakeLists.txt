cmake_minimum_required(VERSION 3.10)

project(skika VERSION 0.0.1)

set(sourcefiles
skika/ensemble/adaptive_random_forest.cpp
skika/ensemble/pearl.cpp
skika/ensemble/lru_state.cpp
skika/ensemble/lossy_state_graph.cpp
)

set(pybindfiles
skika/ensemble/py_binding.cpp
)

set(include_dirs
third_party/streamDM/
)

set(glog "-lglog -lpthread")
set(warning "-Wall -Wextra -Wno-sign-compare -Wno-nonnull -Wno-unused-variable")
set(flags "-fPIC -std=c++14 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O3 -DUNIX -g")

add_definitions(${flags})

add_subdirectory(third_party/streamDM)

message("PEARL: turn on Python Wrapper")
add_subdirectory(third_party/pybind11)
pybind11_add_module(ensemble SHARED ${sourcefiles} ${pybindfiles})

target_link_libraries(ensemble PUBLIC streamDM)
target_include_directories(ensemble PUBLIC ${include_dirs})
