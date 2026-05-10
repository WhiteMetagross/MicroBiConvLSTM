idf_component_register(
    SRCS
        "main.cpp"
        "edge_model_data.cc"
    INCLUDE_DIRS "."
    REQUIRES esp_timer esp_driver_uart
)

if(NOT DEFINED EDGE_TENSOR_ARENA_SIZE)
    set(EDGE_TENSOR_ARENA_SIZE 131072)
endif()

if(NOT DEFINED EDGE_USE_STATIC_ARENA)
    set(EDGE_USE_STATIC_ARENA 0)
endif()

target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN -O3)
target_compile_definitions(
    ${COMPONENT_LIB}
    PRIVATE
        EDGE_TENSOR_ARENA_SIZE=${EDGE_TENSOR_ARENA_SIZE}
        EDGE_USE_STATIC_ARENA=${EDGE_USE_STATIC_ARENA}
)
