import os
import csv
import chardet
from pathlib import Path
import re


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result["encoding"]


def contains_malformed_chars_in_horse_name(file_path, encoding=None):
    """检查CSV文件中的马名列是否包含乱码字符"""
    if encoding is None:
        encoding = detect_encoding(file_path)

    try:
        with open(file_path, "r", encoding=encoding) as file:
            content = file.read()

            # 检查是否包含Ξ或Ѕ字符
            if "Ξ" in content or "Ѕ" in content:
                return True, encoding

            # 尝试解析CSV内容，检查马名列
            file.seek(0)  # 重新定位到文件开头
            reader = csv.DictReader(file)

            for row in reader:
                # 检查马名列（"馬名"列）是否包含乱码字符
                if "馬名" in row:
                    horse_name = row["馬名"]
                    if "Ξ" in horse_name or "Ѕ" in horse_name:
                        return True, encoding

    except UnicodeDecodeError:
        # 如果解码失败，也认为是乱码问题
        return True, encoding
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None, encoding

    return False, encoding


def find_malformed_files(data_dir):
    """查找包含乱码的CSV文件"""
    data_path = Path(data_dir)
    csv_files = list(data_path.rglob("*.csv"))

    malformed_files = []

    print(f"正在扫描 {len(csv_files)} 个CSV文件...")

    for i, csv_file in enumerate(csv_files, 1):
        print(f"检查进度: {i}/{len(csv_files)} - {csv_file.name}")

        is_malformed, detected_encoding = contains_malformed_chars_in_horse_name(csv_file)

        if is_malformed:
            malformed_files.append((csv_file, detected_encoding))
            print(f"  -> 发现乱码文件: {csv_file} (检测编码: {detected_encoding})")

    return malformed_files


def delete_files(file_list):
    """删除文件列表中的文件"""
    deleted_count = 0
    for file_path, encoding in file_list:
        try:
            file_path.unlink()  # 删除文件
            print(f"  已删除: {file_path}")
            deleted_count += 1
        except OSError as e:
            print(f"  删除失败 {file_path}: {e}")

    return deleted_count


def main():
    # 设置数据目录
    DATA_DIR = "d:\\code\\DataScienceIntro\\大作业\\data"

    print("开始查找包含乱码的CSV文件...")
    malformed_files = find_malformed_files(DATA_DIR)

    print(f"\n扫描完成!")
    print(f"包含乱码的文件总数: {len(malformed_files)}")

    if malformed_files:
        print(f"\n包含乱码的文件列表:")
        for file_path, encoding in malformed_files:
            print(f"  - {file_path} (检测编码: {encoding})")

        # 确认删除操作
        response = input(f"\n确认删除这 {len(malformed_files)} 个乱码文件吗? (输入 'yes' 确认删除): ")
        if response.lower() == "yes":
            deleted_count = delete_files(malformed_files)
            print(f"\n删除完成! 共删除了 {deleted_count} 个文件")
        else:
            print("取消删除操作")
    else:
        print("未发现包含乱码的文件")


if __name__ == "__main__":
    main()
