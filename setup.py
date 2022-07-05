from skbuild import setup


setup(
    cmake_args=[
        "-DBUILD_DOCS=OFF",
        "-DBUILD_TESTING=OFF",
    ],
    cmake_install_dir="src/py4dgeo",
)
