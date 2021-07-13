add_subdirectory(c++)

# Copy the data directory into the build tree in order to be able to access it
# with relative paths from tests
file(COPY data DESTINATION .)
