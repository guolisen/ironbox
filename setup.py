from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ironbox",
    version="0.1.0",
    description="Multi-agent platform using LangGraph",
    author="Lewis Guo",
    author_email="info@ironbox.example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Systems Administration",
    ],
    entry_points={
        "console_scripts": [
            "ironbox=ironbox.api.server:main",
            "ironbox-ui=ironbox.ui.app:main",
        ],
    },
)
