cmake_minimum_required(VERSION 3.24)

add_executable(sentinel_train_tiny train_tiny.cpp)
sentinel_apply_common_exe_settings(sentinel_train_tiny)

add_executable(sentinel_generate generate.cpp)
sentinel_apply_common_exe_settings(sentinel_generate)

if(SENTINEL_INSTALL AND NOT DEFINED SKBUILD)
    install(TARGETS sentinel_train_tiny sentinel_generate
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
