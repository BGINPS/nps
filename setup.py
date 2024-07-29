from setuptools import setup


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]


with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()


setup(
    name="nps",
    version="1.0",
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    packages=["romeo"],
    author="Jiwang, Hailin Pan, Qianhua Zhu,",
    author_email="wangji1@genomics.cn",
    description="NPS is a general toolkit for processing protein electrical signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD 2-Clause License",
    url="https://github.com/BGINPS/nps",
)
