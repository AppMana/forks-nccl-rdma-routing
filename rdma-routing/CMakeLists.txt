# libnccl-net-rdmaroute - reachability-aware RDMA device selection for NCCL.
# Standalone NCCL net plugin (loaded via NCCL_NET_PLUGIN); NOT vendored into the
# NCCL tree. This CMake builds + unit-tests the pure routing core; the plugin
# .so (wrapping the v* net ABI) is built on top of it.
cmake_minimum_required(VERSION 3.20)
project(rdmaroute C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pure routing logic (device-agnostic, GID-/64 subnet sensing + conf file).
add_library(rdmaroute STATIC src/route.c src/cts.c)
target_include_directories(rdmaroute PUBLIC src)

# The loadable NCCL net plugin: libnccl-net.so (NCCL_NET_PLUGIN). Standard verbs
# RC backend (ib_net.c) routed per-connection by rdmaroute. Needs libibverbs.
option(RDMAROUTE_PLUGIN "Build the libnccl-net.so plugin" ON)
if(RDMAROUTE_PLUGIN)
  add_library(nccl-net SHARED src/plugin.cc src/ib_net.c src/route.c src/cts.c)
  set_target_properties(nccl-net PROPERTIES OUTPUT_NAME nccl-net PREFIX lib)
  target_include_directories(nccl-net PRIVATE src nccl)
  target_link_libraries(nccl-net PRIVATE ibverbs pthread)
endif()

option(RDMAROUTE_TESTS "Build gtest unit tests" ON)
if(RDMAROUTE_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(route_test tests/route_test.cc)
  target_link_libraries(route_test PRIVATE rdmaroute GTest::gtest_main)
  add_executable(cts_test tests/cts_test.cc)
  target_link_libraries(cts_test PRIVATE rdmaroute GTest::gtest_main)
  add_executable(subnet_match_test tests/subnet_match_test.cc)
  target_include_directories(subnet_match_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/transport/net_ib)
  target_link_libraries(subnet_match_test PRIVATE GTest::gtest_main)
  # Explicit inter-node topology graph (NCCL_NODE_TO_NODE_TOPO_FILE), the
  # declared replacement for gidSameSubnet reachability inference.
  add_executable(node_topo_test tests/node_topo_test.cc)
  target_include_directories(node_topo_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/transport/net_ib)
  target_compile_definitions(node_topo_test PRIVATE
    NODE_TOPO_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/node_topo_example.json")
  target_link_libraries(node_topo_test PRIVATE GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(route_test)
  gtest_discover_tests(cts_test)
  gtest_discover_tests(subnet_match_test)
  gtest_discover_tests(node_topo_test)
endif()
