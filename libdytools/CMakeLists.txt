add_library(
        libdytools

        src/io.cpp

        src/algorithms/dependency-parser.cpp
        src/algorithms/tagger.cpp

        src/builders/builder.cpp
        src/builders/biaffine.cpp
        src/builders/bilstm.cpp
        src/builders/tagger.cpp
        src/builders/token_embeddings.cpp
        src/builders/char_embeddings.cpp
        src/builders/embeddings.cpp

        src/data/conll.cpp

        src/networks/dependency.cpp

        src/loss/dependency.cpp
)

target_include_directories(
        libdytools PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE src
)

FIND_PACKAGE(Boost COMPONENTS serialization filesystem REQUIRED)
target_link_libraries(libdytools ${Boost_LIBRARIES})