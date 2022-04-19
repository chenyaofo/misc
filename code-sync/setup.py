from setuptools import setup

setup(
    name = "pycodesync",
    version = "0.1.2",
    description="Code sync command line application.",
    author="chenyaofo",
    author_email = "chenyaofo@gmail.com",
    packages = ["pycodesync"],
    entry_points = {
        "console_scripts": [
            "csync = pycodesync.__main__:main"
        ]
    },
    install_requires=["tqdm","paramiko"]
)