from skbuild import setup


setup(
    packages=["py4dgeo"],
    package_dir={"": "src"},
    zip_safe=False,
    cmake_args=[
        "-DBUILD_DOCS=OFF",
        "-DBUILD_TESTING=OFF",
    ],
    cmake_install_dir="src/py4dgeo",
)
