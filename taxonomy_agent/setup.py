from setuptools import setup, find_packages

setup(
    name="taxonomy-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers>=2.7.0",
        "hnswlib>=0.8.0",
        "numpy>=1.26.0",
        "pypdf2>=3.0.0",
        "python-docx>=1.1.0",
        "pillow>=10.0.0",
        "anthropic>=0.25.0",
        "watchdog>=4.0.0",
        "pyyaml>=6.0.1",
        "rich>=13.0.0",
        "click>=8.1.7",
    ],
    entry_points={
        "console_scripts": [
            "taxonomy-agent=taxonomy_agent.cli.main:cli",
        ],
    },
    python_requires=">=3.10",
)
