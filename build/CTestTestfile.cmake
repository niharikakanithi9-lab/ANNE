# CMake generated Testfile for 
# Source directory: /home/nihar/ann-engine
# Build directory: /home/nihar/ann-engine/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[all_tests]=] "/home/nihar/ann-engine/build/tests")
set_tests_properties([=[all_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/nihar/ann-engine/CMakeLists.txt;80;add_test;/home/nihar/ann-engine/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
subdirs("_deps/benchmark-build")
subdirs("_deps/cli11-build")
