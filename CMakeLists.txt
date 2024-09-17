# -----------------------------------------------------------------------------
# Project definition
# -----------------------------------------------------------------------------
PROJECT(TLDDC)
cmake_minimum_required (VERSION 3.1)
cmake_policy(SET CMP0057 NEW)
# -----------------------------------------------------------------------------
# CPP11
# -----------------------------------------------------------------------------
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED TRUE)
# -----------------------------------------------------------------------------
# DGtal dependencies
# -----------------------------------------------------------------------------
FIND_PACKAGE(DGtal 1.4 REQUIRED)
INCLUDE_DIRECTORIES(${DGTAL_INCLUDE_DIRS})
LINK_DIRECTORIES(${DGTAL_LIBRARY_DIRS})
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
message(STATUS "DGtal found.")
# -----------------------------------------------------------------------------
# CLI11 et CSV
# -----------------------------------------------------------------------------
include_directories( "${PROJECT_SOURCE_DIR}/ext/" )

find_package(PCL 1.2 REQUIRED COMPONENTS common segmentation)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET(PCLLib ${PCL_SEGMENTATION_LIBRARIES})

ADD_EXECUTABLE(imProfile MainImProfile)
TARGET_LINK_LIBRARIES(imProfile ${DGTAL_LIBRARIES} ${DGtalToolsLibDependencies} ${PCLLib})

