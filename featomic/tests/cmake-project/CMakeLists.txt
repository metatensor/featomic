cmake_minimum_required(VERSION 3.22)

project(featomic-test-cmake-project C CXX)

# We need to update the REQUIRED_FEATOMIC_VERSION in the same way we update the
# featomic version for dev builds
include(../../cmake/dev-versions.cmake)
set(REQUIRED_FEATOMIC_VERSION "0.6.4")
create_development_version("${REQUIRED_FEATOMIC_VERSION}" FEATOMIC_FULL_VERSION "featomic-v")
string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_FEATOMIC_VERSION ${FEATOMIC_FULL_VERSION})
find_package(featomic ${REQUIRED_FEATOMIC_VERSION} REQUIRED)

add_executable(c-main main.c)
target_link_libraries(c-main featomic::shared)

add_executable(c-main-static main.c)
target_link_libraries(c-main-static featomic::static)

add_executable(cxx-main main.cpp)
target_link_libraries(cxx-main featomic::shared)

add_executable(cxx-main-static main.cpp)
target_link_libraries(cxx-main-static featomic::static)
