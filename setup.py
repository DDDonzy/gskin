
import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# --- 配置区 ---
# Maya 和 OpenCL 的路径，请根据您的系统进行修改
MAYA_LOCATION = os.environ.get("MAYA_LOCATION", "/usr/autodesk/maya2024")
# OpenCL 路径可能需要调整，特别是如果您使用的是NVIDIA CUDA SDK或AMD APP SDK
OPENCL_LOCATION = os.environ.get("OPENCL_LOCATION", "/usr/local/cuda") 

# --- 平台特定的编译设置 ---
if sys.platform == "win32":
    include_dirs = [
        os.path.join(MAYA_LOCATION, "include"),
        numpy.get_include(),
        os.path.join(OPENCL_LOCATION, "include"),
    ]
    library_dirs = [
        os.path.join(MAYA_LOCATION, "lib"),
        os.path.join(OPENCL_LOCATION, "lib/x64"),
    ]
    libraries = ["OpenMaya", "Foundation", "OpenCL"]
    extra_compile_args=["/openmp"]
    extra_link_args=["/openmp"]

else: # Linux & macOS
    include_dirs = [
        os.path.join(MAYA_LOCATION, "include"),
        numpy.get_include(),
        os.path.join(OPENCL_LOCATION, "include"), # Linux
    ]
    library_dirs = [
        os.path.join(MAYA_LOCATION, "lib"),
        os.path.join(OPENCL_LOCATION, "lib64"), # Linux
    ]
    libraries = ["OpenMaya", "Foundation", "OpenCL"]
    extra_compile_args=["-fopenmp", "-std=c++11"]
    extra_link_args=["-fopenmp"]

# --- 查找所有要编译的 Cython 文件 ---
extensions = []
for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".pyx"):
            module_path = os.path.join(root, file)
            module_name = os.path.splitext(module_path)[0].replace(os.path.sep, ".")
            
            print(f"Found Cython file: {module_path} -> Module: {module_name}")
            
            ext = Extension(
                name=module_name,
                sources=[module_path],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
            extensions.append(ext)

# --- 执行编译 ---
if not extensions:
    print("No Cython .pyx files found to compile. Exiting.")
else:
    setup(
        name="zNPSpeed",
        ext_modules=cythonize(
            extensions,
            compiler_directives={'language_level': "3"},
        ),
        # build_ext --inplace 会将编译好的 .pyd/.so 文件放在源文件旁边
        script_args=["build_ext", "--inplace"]
    )
    print("\nCompilation finished. .pyd/.so files should be next to their .pyx sources.")
