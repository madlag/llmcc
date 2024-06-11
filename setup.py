from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="llmcc",
    version="0.1",
    description='llmcc is a python package for easy use of Large Language Models, exposing them as "compilers" from prompt to output.',
    long_description=readme(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.0",
        "Topic :: Text Processing",
    ],
    keywords="",
    url="",
    author="",
    author_email="",
    license="MIT",
    packages=["llmcc"],
    install_requires=["click", "jinja2", "openai", "anthropic"],
    test_suite="nose.collector",
    tests_require=["nose", "nose-cover3"],
    entry_points={
        "console_scripts": ["llmcc=llmcc.llmcc:main"],
    },
    include_package_data=True,
    zip_safe=False,
)
