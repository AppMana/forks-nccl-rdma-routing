# Unit tests for src/transport/net_ib/{subnet_match.h,node_topo.h}
cmake_minimum_required(VERSION 3.20)
project(nccl_rdma_routing_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(subnet_match_test tests/subnet_match_test.cc)
target_include_directories(subnet_match_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/transport/net_ib)
target_link_libraries(subnet_match_test PRIVATE GTest::gtest_main)

add_executable(rail_select_test tests/rail_select_test.cc)
target_include_directories(rail_select_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/transport/net_ib)
target_link_libraries(rail_select_test PRIVATE GTest::gtest_main)

add_executable(node_topo_test tests/node_topo_test.cc)
target_include_directories(node_topo_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/transport/net_ib)
target_compile_definitions(node_topo_test PRIVATE
  NODE_TOPO_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/node_topo_example.json")
target_link_libraries(node_topo_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(subnet_match_test)
gtest_discover_tests(rail_select_test)
gtest_discover_tests(node_topo_test)
