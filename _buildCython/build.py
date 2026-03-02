import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# ==================== 1. 核心路径定义 ====================
# 源码目录 (绝对路径)
SRC_DIR = Path(r"E:\d_maya\gskin\src").resolve()
# 当前脚本路径
CURRENT_SCRIPT = Path(__file__).resolve()


def get_compiler_python():
    exe_path = Path(sys.executable)
    if "maya.exe" in exe_path.name.lower():
        mayapy = exe_path.with_name("mayapy.exe")
        return str(mayapy) if mayapy.exists() else str(exe_path)
    return str(exe_path)


# ==================== 2. 自动化流水线函数 ====================


def get_relative_files(folder, pattern, exclude_list=None):
    """获取相对于 folder 的文件名列表"""
    if exclude_list is None:
        exclude_list = []
    files = list(folder.glob(pattern))
    return [f.name for f in files if f.name not in exclude_list]


def create_extensions(file_names, **ext_kwargs):
    """将相对路径文件名转化为 Extension 对象"""
    extensions = []
    for file_name in file_names:
        p = Path(file_name)
        ext = Extension(name=p.stem, sources=[file_name], **ext_kwargs)
        extensions.append(ext)
    return extensions


def cleanup_src():
    """在 SRC_DIR 目录下精准打扫卫生 (已修复安全隐患)"""
    print(f"\n🧹 [清理] 正在清理源码目录: {SRC_DIR}")

    # 删除源码目录下的 build 文件夹
    build_dir = SRC_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # 【安全清理逻辑】：只删除与 .pyx .py 同名的 .c / .cpp / .html 文件
    for pyx_file in list(SRC_DIR.glob("*.pyx")) + list(SRC_DIR.glob("*.py")):
        generated_files = [pyx_file.with_suffix(".c"), pyx_file.with_suffix(".cpp"), pyx_file.with_suffix(".html")]
        for f in generated_files:
            if f.exists():
                try:
                    os.remove(f)
                except Exception as e:
                    pass
    print("✨ [完成] 目录已清理干净。")


# ==================== 3. 编译配置 ====================

if sys.platform.startswith("win"):
    compile_args = ["/openmp", "/O2"]
    link_args = ["/openmp"]
else:
    compile_args = ["-fopenmp", "-O3"]
    link_args = ["-fopenmp"]

shared_config = {
    "extra_compile_args": compile_args,
    "extra_link_args": link_args,
    "include_dirs": [np.get_include(), r"C:\Users\Donzy\Downloads\Autodesk_Maya_2024_2_Update_DEVKIT_Windows\devkitBase\include\Python310\Python"],
    "library_dirs": [r"C:\Program Files\Autodesk\Maya2024\lib"],
    # ==========================================
    # 👇 【关键修改】：强制 Cython 使用 C++ 编译器
    # ==========================================
    "language": "c++",
}

# ==================== 4. 执行控制 ====================

if __name__ == "__main__":
    # 状态 A：正式编译阶段
    if "build_ext" in sys.argv:
        pyx_files = get_relative_files(Path("."), "*.pyx")
        py_files = get_relative_files(Path("."), "*Cython.py", exclude_list=[CURRENT_SCRIPT.name])

        # 这里的 shared_config 中的 language="c++" 会被完美传递给 Extension
        all_exts = create_extensions(pyx_files + py_files, **shared_config)

        setup(ext_modules=cythonize(all_exts, compiler_directives={"language_level": "3"}), script_args=sys.argv[1:])

    # 状态 B：启动/自举阶段
    else:
        print(f"🚀 [启动] 锁定目标目录: {SRC_DIR}")
        compiler = get_compiler_python()
        cmd = [compiler, str(CURRENT_SCRIPT), "build_ext", "--inplace"]

        try:
            subprocess.run(cmd, check=True, cwd=str(SRC_DIR))
            print(f"\n✅ [成功] 编译产物已安全存放在: {SRC_DIR}")
            cleanup_src()
        except subprocess.CalledProcessError:
            print("\n❌ [失败] 编译过程中断，请检查上方红字报错。")
