from setuptools import setup, find_packages

setup(
    name="nt_ops",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "vllm.general_plugins": [
            "nt_ops_mlu = nt_ops:register_nt_ops_mlu_plugin",
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "ninetoothed>=0.16.0",
    ],
)
