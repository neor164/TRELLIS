import setuptools
import subprocess
import sys
import os
import toml

# ... (rest of your functions: get_torch_version, get_cuda_version, etc.) ...


# Read pyproject.toml for base dependencies
with open("pyproject.toml", "r") as f:
    config = toml.load(f)
    install_requires = config['project']['dependencies']


extras_require = {
    'basic': config['project']['optional-dependencies']['basic'],
    'xformers': config['project']['optional-dependencies']['xformers'],
    'flash-attn': config['project']['optional-dependencies']['flash_attn'],
    'diffoctreerast': config['project']['optional-dependencies']['diffoctreerast'],
    'vox2seq': config['project']['optional-dependencies']['vox2seq'],
    'spconv': config['project']['optional-dependencies']['spconv'],
    'mipgaussian': config['project']['optional-dependencies']['mipgaussian'],
    'kaolin': config['project']['optional-dependencies']['kaolin'],
    'nvdiffrast': config['project']['optional-dependencies']['nvdiffrast'],
    'demo': config['project']['optional-dependencies']['demo'],
}

# Filter out empty lists from extras_require
filtered_extras_require = {}
for key, value in extras_require.items():
    if value:  # Check if the list is not empty
        filtered_extras_require[key] = value


package_data = {
    'trellis': ['README.md'], # if you have any package data
}


setuptools.setup(
    name="trellis", # same name as in pyproject.toml
    version="0.1.0", # same version as in pyproject.toml
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require=filtered_extras_require, # Use the filtered dictionary
    python_requires=">=3.10, <=3.12", # same as in pyproject.toml
    package_data=package_data,
    entry_points={}, # if you have any scripts or entry points
)

# ... (rest of your post-install script logic) ...