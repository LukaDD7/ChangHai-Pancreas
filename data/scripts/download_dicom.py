#!/usr/bin/env python
"""
下载CPTAC-PDA DICOM数据
使用TCIA Utils库
"""

from tcia_utils import nbia
import os

# 设置下载目录
download_dir = "/media/luzhenyang/project/ChangHai_PDA/data/dicom_downloads"
os.makedirs(download_dir, exist_ok=True)

# 从manifest文件中提取Series UID
series_uid = "1.3.6.1.4.1.14519.5.2.1.1078.3707.326044378573382028419735871265"

print(f"开始下载CPTAC-PDA数据...")
print(f"Series UID: {series_uid}")
print(f"下载目录: {download_dir}")
print()

# 下载DICOM数据
# 方法1: 使用series UID下载
nbia.downloadSeries([series_uid], path=download_dir)

print("\n下载完成!")
print(f"数据保存在: {download_dir}")
