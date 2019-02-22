add_library(
        libdytools

        src/io.cpp
        src/dict.cpp
        src/utils.cpp
        src/masked_sequence.cpp

        src/algorithms/dependency-parser.cpp
        src/algorithms/span-parser.cpp
        src/algorithms/tagger.cpp

        src/builders/biaffine.cpp
        src/builders/biaffine_tagger.cpp
        src/builders/bilstm.cpp
        src/builders/tagger.cpp
        src/builders/embeddings/word.cpp
        src/builders/embeddings/character.cpp
        src/builders/embeddings/embeddings.cpp
        src/builders/gcn.cpp
        src/builders/mlp.cpp
        src/builders/masked_lstm.cpp
        src/builders/masked_bilstm.cpp

        src/data/conll.cpp

        src/networks/dependency.cpp
        src/networks/base-dependency.cpp

        src/loss/dependency.cpp

        src/functions/root_arborescence_marginals.cpp
        src/functions/masking.cpp
)

FIND_PACKAGE(Boost COMPONENTS regex serialization filesystem REQUIRED)
target_link_libraries(libdytools ${Boost_LIBRARIES})

target_include_directories(
        libdytools PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE src
)