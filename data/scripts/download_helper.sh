#!/bin/bash
# 使用TCIA Data Retriever命令行工具下载CPTAC-PDA数据

# 配置
MANIFEST_FILE="/media/luzhenyang/project/ChangHai_PDA/data/manifest-1773998032343.tcia"
OUTPUT_DIR="/media/luzhenyang/project/ChangHai_PDA/data/dicom_data"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "CPTAC-PDA DICOM 下载脚本"
echo "================================================"
echo ""
echo "由于网络SSL连接问题，请使用以下方法之一："
echo ""
echo "方法1: 使用TCIA官方Java下载器"
echo "  1. 下载: https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever/NBIADataRetriever_4.4.zip"
echo "  2. 解压并运行:"
echo "     java -jar NBIADataRetriever.jar -c $MANIFEST_FILE -o $OUTPUT_DIR -v"
echo ""
echo "方法2: 浏览器手动下载"
echo "  1. 访问: https://nbia.cancerimagingarchive.net/"
echo "  2. 登录账号"
echo "  3. 搜索患者ID（如 C3N-00302）"
echo "  4. 下载CT序列"
echo ""
echo "方法3: 使用curl（已尝试，可能因SSL失败）"
echo "  curl -L -o output.zip 'https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet?seriesuids=...'"
echo ""
echo "================================================"
echo ""
echo "当前manifest文件包含的Series UID:"
grep "^[0-9]" "$MANIFEST_FILE"
