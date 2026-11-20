"""把插件打包成可安装的 zip（Sakura 设置 → 插件 → 从 ZIP 安装）。

    python build_zip.py

打出来的 zip 里只有一个顶层目录，名字就是插件 id：Sakura 的安装器靠它找 plugin.yaml。
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PACKAGE_DIR_NAME = "sakuratools.emotion_pool"
ROOT_FILES = ("plugin.yaml", "plugin.py", "selftest.py", "README.md")
PACKAGE_DIR = "emotion_pool"
FRONTEND_DIR = "frontend"
SKIP_PARTS = {"__pycache__", ".selftest-tmp", ".git"}


def version() -> str:
    text = (HERE / "plugin.yaml").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", text, re.M)
    return match.group(1) if match else "0.0.0"


def collect() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for name in ROOT_FILES:
        path = HERE / name
        if path.is_file():
            files.append((path, name))
    package_root = HERE / PACKAGE_DIR
    if package_root.is_dir():
        for path in sorted(package_root.rglob("*.py")):
            if SKIP_PARTS & set(path.parts):
                continue
            files.append((path, path.relative_to(HERE).as_posix()))
    # 仪表盘的自定义前端模块：宿主把这段 JS 当 ES module 导入，所以必须一起打包
    frontend_root = HERE / FRONTEND_DIR
    if frontend_root.is_dir():
        for path in sorted(frontend_root.rglob("*.js")):
            if SKIP_PARTS & set(path.parts):
                continue
            files.append((path, path.relative_to(HERE).as_posix()))
    return files


def main() -> int:
    files = collect()
    names = [relative for _path, relative in files]
    missing = [name for name in ("plugin.yaml", "plugin.py") if name not in names]
    if missing:
        print("缺少必需文件：" + "、".join(missing))
        return 1
    target = HERE / f"{PACKAGE_DIR_NAME}-{version()}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in files:
            archive.write(path, arcname=f"{PACKAGE_DIR_NAME}/{relative}")
    print(f"打包完成：{target}")
    print(f"文件数：{len(files)}，体积：{max(1, target.stat().st_size // 1024)} KB")
    for relative in names:
        print("  " + relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
