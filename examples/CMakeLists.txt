add_executable(inference inference.cpp)
target_link_libraries(inference RNLTLib ${TORCH_LIBRARIES})

add_executable(train train.cpp)
target_link_libraries(train RNLTLib ${TORCH_LIBRARIES} -lboost_program_options)
