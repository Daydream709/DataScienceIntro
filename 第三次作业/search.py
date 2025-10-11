import json
import os

output_dir = "ecnu_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ecnu_names = ["EAST CHINA NORMAL UNIVERSITY", "East China Normal University", "华东师范大学"]


data_dir = "data"
results = {}

print("开始搜索")

for filename in os.listdir(data_dir):
    if filename.endswith(".json") and filename != "esi_data.json":
        file_path = os.path.join(data_dir, filename)


        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"无法解析文件: {filename}")
                continue


        field_name = filename.replace(".json", "")
        results[field_name] = []

        if "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if isinstance(item, dict) and "institution" in item:
                    institution_name = item["institution"]

                    for name in ecnu_names:
                        if (
                            name.upper() in institution_name.upper()
                            or institution_name.upper() in name.upper()
                        ):
                            results[field_name].append(item)
                            break


        if results[field_name]:
            print(f"在 {field_name} 领域找到 {len(results[field_name])} 条相关记录")
        else:
            del results[field_name]


if results:
    output_file = os.path.join(output_dir, "ecnu_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n总共在 {len(results)} 个领域找到华东师范大学的相关信息")
    print(f"结果已保存到: {output_file}")

else:
    print("未找到华东师范大学的相关信息")
