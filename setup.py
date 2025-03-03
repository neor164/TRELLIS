import setuptools
import subprocess
import sys
import os
import toml
import argparse

def get_torch_version():
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None

def get_cuda_version():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
        else:
            return None
    except ImportError:
        return None

def get_hip_version():
    try:
        import torch
        if torch.cuda.is_available() and torch.version.hip: # check torch.cuda.is_available() first to avoid error on non-cuda builds
            return torch.version.hip
        else:
            return None
    except ImportError:
        return None

def install_from_git(url, dest):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'git+{url}', '-t', dest])

def install_nvdiffrast_uv():
    # nvdiffrast is already handled by uv in pyproject.toml
    pass # No additional install needed here for nvdiffrast as it's in tool.uv.sources

def install_diffoctreerast():
    if PLATFORM == "cuda":
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/JeffreyXiang/diffoctreerast.git'])
    else:
        print("[DIFFOCTREERAST] Unsupported platform: {}".format(PLATFORM))

def install_mipgaussian():
    if PLATFORM == "cuda":
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization'])
    else:
        print("[MIPGAUSSIAN] Unsupported platform: {}".format(PLATFORM))

def install_vox2seq():
    if PLATFORM == "cuda":
        # Assuming 'extensions/vox2seq' is in your project directory
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'extensions/vox2seq']) # adjust path if needed
    else:
        print("[VOX2SEQ] Unsupported platform: {}".format(PLATFORM))

def install_spconv():
    if PLATFORM == "cuda":
        cuda_version = get_cuda_version()
        cuda_major_version = cuda_version.split('.')[0] if cuda_version else 'Unknown'
        if cuda_major_version == "11":
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'spconv-cu118'])
        elif cuda_major_version == "12":
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'spconv-cu120'])
        else:
            print("[SPCONV] Unsupported PyTorch CUDA version: {}".format(cuda_major_version))
    else:
        print("[SPCONV] Unsupported platform: {}".format(PLATFORM))

def install_kaolin():
    if PLATFORM == "cuda":
        torch_version = get_torch_version()
        if torch_version:
            if torch_version.startswith("2.0.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaolin', '-f', 'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html'])
            elif torch_version.startswith("2.1.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaolin', '-f', 'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html'])
            elif torch_version.startswith("2.1.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaolin', '-f', 'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html'])
            elif torch_version.startswith("2.2.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaolin', '-f', 'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.0_cu118.html'])
            elif torch_version.startswith("2.2.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaolin', '-f', 'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.1_cu118.html'])
            elif torch_version.startswith("2.2.2"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaolin', '-f', 'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu118.html'])
            elif torch_version.startswith("2.4.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaolin', '-f', 'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html']) # using cu121 for 2.4.0 as per script, might need to double check
            else:
                print("[KAOLIN] Unsupported PyTorch version: {}".format(torch_version))
        else:
            print("[KAOLIN] PyTorch version could not be determined.")
    else:
        print("[KAOLIN] Unsupported platform: {}".format(PLATFORM))


def install_xformers():
    if PLATFORM == "cuda":
        cuda_version = get_cuda_version()
        torch_version = get_torch_version()
        if cuda_version == "11.8":
            if torch_version and torch_version.startswith("2.0.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'https://files.pythonhosted.org/packages/52/ca/82aeee5dcc24a3429ff5de65cc58ae9695f90f49fbba71755e7fab69a706/xformers-0.0.22-cp310-cp310-manylinux2014_x86_64.whl'])
            elif torch_version and torch_version.startswith("2.1.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.22.post7', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            elif torch_version and torch_version.startswith("2.1.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.23', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            elif torch_version and torch_version.startswith("2.1.2"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.23.post1', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            elif torch_version and torch_version.startswith("2.2.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.24', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            elif torch_version and torch_version.startswith("2.2.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.25', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            elif torch_version and torch_version.startswith("2.2.2"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.25.post1', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            elif torch_version and torch_version.startswith("2.3.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.26.post1', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            elif torch_version and torch_version.startswith("2.4.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.27.post2', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            elif torch_version and torch_version.startswith("2.4.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.28', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            elif torch_version and torch_version.startswith("2.5.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.28.post2', '--index-url', 'https://download.pytorch.org/whl/cu118'])
            else:
                print("[XFORMERS] Unsupported PyTorch & CUDA version: {} & {}".format(torch_version, cuda_version))
        elif cuda_version == "12.1":
            if torch_version and torch_version.startswith("2.1.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.22.post7', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            elif torch_version and torch_version.startswith("2.1.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.23', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            elif torch_version and torch_version.startswith("2.1.2"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.23.post1', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            elif torch_version and torch_version.startswith("2.2.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.24', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            elif torch_version and torch_version.startswith("2.2.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.25', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            elif torch_version and torch_version.startswith("2.2.2"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.25.post1', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            elif torch_version and torch_version.startswith("2.3.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.26.post1', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            elif torch_version and torch_version.startswith("2.4.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.27.post2', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            elif torch_version and torch_version.startswith("2.4.1"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.28', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            elif torch_version and torch_version.startswith("2.5.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.28.post2', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            else:
                print("[XFORMERS] Unsupported PyTorch & CUDA version: {} & {}".format(torch_version, cuda_version))
        elif cuda_version == "12.4":
            if torch_version and torch_version.startswith("2.5.0"):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.28.post2', '--index-url', 'https://download.pytorch.org/whl/cu124'])
            else:
                print("[XFORMERS] Unsupported PyTorch & CUDA version: {} & {}".format(torch_version, cuda_version))
        else:
            cuda_major_version = cuda_version.split('.')[0] if cuda_version else 'Unknown'
            print("[XFORMERS] Unsupported CUDA version: {}".format(cuda_major_version))
    elif PLATFORM == "hip":
        torch_version = get_torch_version()
        if torch_version and torch_version.startswith("2.4.1+rocm6.1"):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xformers==0.0.28', '--index-url', 'https://download.pytorch.org/whl/rocm6.1'])
        else:
            print("[XFORMERS] Unsupported PyTorch version: {}".format(torch_version))
    else:
        print("[XFORMERS] Unsupported platform: {}".format(PLATFORM))


def install_flash_attn():
    if PLATFORM == "cuda":
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flash-attn'])
    elif PLATFORM == "hip":
        print("[FLASHATTN] Prebuilt binaries not found. Building from source...")
        extensions_dir = '/tmp/extensions'
        os.makedirs(extensions_dir, exist_ok=True)
        flash_attn_dir = os.path.join(extensions_dir, 'flash-attention')
        if not os.path.exists(flash_attn_dir):
            subprocess.check_call(['git', 'clone', '--recursive', 'https://github.com/ROCm/flash-attention.git', flash_attn_dir])
        subprocess.check_call(['git', 'checkout', 'tags/v2.6.3-cktile'], cwd=flash_attn_dir)
        subprocess.check_call(['GPU_ARCHS=gfx942', 'python', 'setup.py', 'install'], cwd=flash_attn_dir) #MI300 series
    else:
        print("[FLASHATTN] Unsupported platform: {}".format(PLATFORM))


# Get system information
PYTORCH_VERSION = get_torch_version()
PLATFORM = "unknown"
CUDA_VERSION = get_cuda_version()
HIP_VERSION = get_hip_version()

if CUDA_VERSION:
    PLATFORM = "cuda"
    CUDA_MAJOR_VERSION = CUDA_VERSION.split('.')[0] if CUDA_VERSION else 'Unknown'
    CUDA_MINOR_VERSION = CUDA_VERSION.split('.')[1] if CUDA_VERSION else 'Unknown'
    print(f"[SYSTEM] PyTorch Version: {PYTORCH_VERSION}, CUDA Version: {CUDA_VERSION}")
elif HIP_VERSION:
    PLATFORM = "hip"
    HIP_MAJOR_VERSION = HIP_VERSION.split('.')[0] if HIP_VERSION else 'Unknown'
    HIP_MINOR_VERSION = HIP_VERSION.split('.')[1] if HIP_VERSION else 'Unknown'
    if PYTORCH_VERSION != "2.4.1+rocm6.1":
        print(f"[SYSTEM] Installing PyTorch 2.4.1 for HIP ({PYTORCH_VERSION} -> 2.4.1+rocm6.1)")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==2.4.1', 'torchvision==0.19.1', '--index-url', 'https://download.pytorch.org/whl/rocm6.1', '--user'])
        extensions_dir = '/tmp/extensions'
        os.makedirs(extensions_dir, exist_ok=True)
        amd_smi_dir = os.path.join(extensions_dir, 'amd_smi')

        if not os.path.exists(amd_smi_dir):
            subprocess.check_call(['sudo', 'cp', '-r', '/opt/rocm/share/amd_smi', amd_smi_dir])
        subprocess.check_call(['sudo', 'chmod', '-R', '777', amd_smi_dir])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'], cwd=amd_smi_dir)

        PYTORCH_VERSION = get_torch_version() # Update PyTorch version after install
    print(f"[SYSTEM] PyTorch Version: {PYTORCH_VERSION}, HIP Version: {HIP_VERSION}")
else:
    PLATFORM = "cpu"


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
    if value:
        filtered_extras_require[key] = value


package_data = {
    'trellis': ['README.md'], # if you have any package data
}


setuptools.setup(
    name="trellis", # same name as in pyproject.toml
    version="0.1.0", # same version as in pyproject.toml
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require=filtered_extras_require,
    python_requires=">=3.10, <=3.12", # same as in pyproject.toml
    package_data=package_data,
    entry_points={}, # if you have any scripts or entry points
)


# post_install script logic (mimicking setup.sh argument handling)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Setup script for trellis project.")
    parser.add_argument("--basic", action="store_true", help="Install basic dependencies")
    parser.add_argument("--xformers", action="store_true", help="Install xformers")
    parser.add_argument("--flash-attn", action="store_true", help="Install flash-attn")
    parser.add_argument("--diffoctreerast", action="store_true", help="Install diffoctreerast")
    parser.add_argument("--vox2seq", action="store_true", help="Install vox2seq")
    parser.add_argument("--spconv", action="store_true", help="Install spconv")
    parser.add_argument("--mipgaussian", action="store_true", help="Install mip-splatting")
    parser.add_argument("--kaolin", action="store_true", help="Install kaolin")
    parser.add_argument("--nvdiffrast", action="store_true", help="Install nvdiffrast")
    parser.add_argument("--demo", action="store_true", help="Install all dependencies for demo")
    parser.add_argument("--new-env", action="store_true", help="Create a new conda environment (handled outside setup.py)") # Inform user about new-env

    args = parser.parse_args()

    if args.new_env:
        print("Warning: --new-env argument is not handled by setup.py. Please use conda or uv directly to create a new environment.")

    if args.basic:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[basic]'])
    if args.xformers:
        install_xformers()
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[xformers]']) # install the extra to mark it as installed
    if args.flash_attn:
        install_flash_attn()
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[flash-attn]'])
    if args.diffoctreerast:
        install_diffoctreerast()
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[diffoctreerast]'])
    if args.vox2seq:
        install_vox2seq()
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[vox2seq]'])
    if args.spconv:
        install_spconv()
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[spconv]'])
    if args.mipgaussian:
        install_mipgaussian()
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[mipgaussian]'])
    if args.kaolin:
        install_kaolin()
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[kaolin]'])
    if args.nvdiffrast:
        install_nvdiffrast_uv() # uv handles nvdiffrast already, this just triggers the extra install
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[nvdiffrast]'])
    if args.demo:
         subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trellis[demo]'])

    if not any([args.basic, args.xformers, args.flash_attn, args.diffoctreerast, args.vox2seq, args.spconv, args.mipgaussian, args.kaolin, args.nvdiffrast, args.demo]):
        print("Usage: python setup.py [--basic] [--xformers] [--flash-attn] [--diffoctreerast] [--vox2seq] [--spconv] [--mipgaussian] [--kaolin] [--nvdiffrast] [--demo]")