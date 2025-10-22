import pymysql
import csv
import os
from collections import defaultdict

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "tyj810915mysql",
    "database": "esi_data",
    "charset": "utf8mb4",
}


def connect_database():
    """连接数据库"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        print("数据库连接成功")
        return connection
    except Exception as e:
        print(f"数据库连接失败: {e}")
        raise


def load_ecnu_indicators():
    """从CSV文件加载华东师范大学的指标"""
    indicators = {}
    filepath = os.path.join("QueryData", "ecnu_indicators.csv")

    try:
        with open(filepath, "r", encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            for row in reader:
                # 特殊处理"进入前500名的学科"指标，它是一个列表
                if row[0] == "进入前500名的学科":
                    # 从分号分隔的字符串还原为列表
                    if row[1]:
                        indicators[row[0]] = row[1].split("; ")
                    else:
                        indicators[row[0]] = []
                else:
                    # 确保所有值都是float类型
                    indicators[row[0]] = float(row[1])
        print("成功加载华东师范大学指标数据")
        return indicators
    except Exception as e:
        print(f"读取指标文件时出错: {e}")
        raise


def calculate_university_indicators(connection, university_id):
    """计算指定大学的各项指标"""
    indicators = {}

    try:
        with connection.cursor() as cursor:
            # 1. 进入 ESI 前 1%的学科数量
            query1 = """
            SELECT COUNT(*) AS 学科总数
            FROM rankings r
            WHERE r.university_id = %s;
            """
            cursor.execute(query1, (university_id,))
            indicators["进入ESI前1%的学科数量"] = int(cursor.fetchone()[0])

            # 2. 各学科排名分布情况（顶尖/中上/中等/其他）
            query2 = """
            SELECT 
                SUM(CASE WHEN r.rank_order <= 100 THEN 1 ELSE 0 END) AS 顶尖学科数,
                SUM(CASE WHEN r.rank_order > 100 AND r.rank_order <= 500 THEN 1 ELSE 0 END) AS 中上学科数,
                SUM(CASE WHEN r.rank_order > 500 AND r.rank_order <= 1000 THEN 1 ELSE 0 END) AS 中等学科数,
                SUM(CASE WHEN r.rank_order > 1000 THEN 1 ELSE 0 END) AS 其他学科数
            FROM rankings r
            WHERE r.university_id = %s;
            """
            cursor.execute(query2, (university_id,))
            distribution = cursor.fetchone()
            indicators["顶尖学科数(≤100)"] = int(distribution[0]) if distribution[0] else 0
            indicators["中上学科数(101-500)"] = int(distribution[1]) if distribution[1] else 0
            indicators["中等学科数(501-1000)"] = int(distribution[2]) if distribution[2] else 0
            indicators["其他学科数(>1000)"] = int(distribution[3]) if distribution[3] else 0

            # 新增指标：获取进入前500名的学科名称
            query_top500_subjects = """
            SELECT s.name
            FROM rankings r
            JOIN subjects s ON r.subject_id = s.id
            WHERE r.university_id = %s
            AND r.rank_order <= 500
            ORDER BY r.rank_order;
            """
            cursor.execute(query_top500_subjects, (university_id,))
            top500_subjects = cursor.fetchall()
            indicators["进入前500名的学科"] = [subject[0] for subject in top500_subjects]

            # 3. 学科领域覆盖范围（理工科、文科、医科等比例）
            query3 = """
            SELECT 
                SUM(CASE 
                    WHEN s.name IN ('ENGINEERING', 'MATERIALS SCIENCE', 'CHEMISTRY', 'COMPUTER SCIENCE', 'MATHEMATICS', 'PHYSICS') 
                    THEN 1 ELSE 0 
                END) AS 理工科,
                SUM(CASE 
                    WHEN s.name IN ('ECONOMICS & BUSINESS', 'SOCIAL SCIENCES, GENERAL', 'PSYCHIATRY_PSYCHOLOGY') 
                    THEN 1 ELSE 0 
                END) AS 文科,
                SUM(CASE 
                    WHEN s.name IN ('CLINICAL MEDICINE', 'IMMUNOLOGY', 'MICROBIOLOGY', 'PHARMACOLOGY & TOXICOLOGY') 
                    THEN 1 ELSE 0 
                END) AS 医科
            FROM rankings r
            JOIN subjects s ON r.subject_id = s.id
            WHERE r.university_id = %s;
            """
            cursor.execute(query3, (university_id,))
            field_coverage = cursor.fetchone()
            indicators["理工科领域学科数"] = int(field_coverage[0]) if field_coverage[0] else 0
            indicators["文科领域学科数"] = int(field_coverage[1]) if field_coverage[1] else 0
            indicators["医科领域学科数"] = int(field_coverage[2]) if field_coverage[2] else 0

            # 4. 研究实力指标
            query4 = """
            SELECT 
                SUM(r.papers) AS 论文总数,
                SUM(r.citations) AS 引用次数总数,
                CASE 
                    WHEN SUM(r.papers) > 0 THEN ROUND(SUM(r.citations) / SUM(r.papers), 2)
                    ELSE 0 
                END AS 平均每篇论文引用次数
            FROM rankings r
            WHERE r.university_id = %s;
            """
            cursor.execute(query4, (university_id,))
            research_metrics = cursor.fetchone()
            indicators["论文总数"] = int(research_metrics[0]) if research_metrics[0] else 0
            indicators["引用次数总数"] = int(research_metrics[1]) if research_metrics[1] else 0
            indicators["平均每篇论文引用次数"] = float(research_metrics[2]) if research_metrics[2] else 0.0

    except Exception as e:
        print(f"分析大学指标过程中出现错误: {e}")
        raise

    return indicators


def get_all_universities(connection):
    """获取所有大学的信息"""
    try:
        with connection.cursor() as cursor:
            query = "SELECT id, name FROM universities"
            cursor.execute(query)
            universities = cursor.fetchall()
            return [{"id": uni[0], "name": uni[1]} for uni in universities]
    except Exception as e:
        print(f"获取大学列表时出错: {e}")
        raise


def calculate_jaccard_similarity(set1, set2):
    """计算两个集合的Jaccard相似度"""
    set1 = set(set1)
    set2 = set(set2)

    if not set1 and not set2:
        return 1.0  # 两个空集被认为是完全相似的

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0

    return intersection / union


def calculate_similarity(ecnu_indicators, uni_indicators):
    """计算两个大学之间的相似度（使用欧氏距离）"""
    # 对于数值型指标，我们计算相对差异
    total_difference = 0
    count = 0

    for key in ecnu_indicators.keys():
        # 特殊处理"进入前500名的学科"指标，使用Jaccard相似度
        if key == "进入前500名的学科":
            ecnu_subjects = ecnu_indicators[key]
            uni_subjects = uni_indicators.get(key, [])
            jaccard_similarity = calculate_jaccard_similarity(ecnu_subjects, uni_subjects)
            # 将相似度转换为差异度（1 - 相似度）
            difference = 1 - jaccard_similarity
        elif key in uni_indicators and uni_indicators[key] is not None:
            # 确保两个值都是float类型，避免decimal和float运算错误
            ecnu_value = float(ecnu_indicators[key])
            uni_value = float(uni_indicators[key])

            # 对于非零指标，计算相对差异
            if ecnu_value != 0:
                difference = abs(ecnu_value - uni_value) / ecnu_value
            else:
                # 如果ECNU的指标为0，则直接比较绝对值差异
                difference = abs(ecnu_value - uni_value)
        else:
            # 如果该指标在比较的大学中不存在，则跳过
            continue

        total_difference += difference
        count += 1

    # 返回相似度分数（差异越小，相似度越高）
    if count > 0:
        avg_difference = total_difference / count
        similarity_score = 1 / (1 + avg_difference)  # 转换为相似度分数
        return similarity_score
    else:
        return 0


def find_similar_universities(connection, ecnu_indicators, top_n=10):
    """查找与华东师范大学最相似的大学"""
    universities = get_all_universities(connection)

    similarities = []

    for university in universities:
        # 跳过华东师范大学本身
        if university["name"] in ["EAST CHINA NORMAL UNIVERSITY", "华东师范大学"]:
            continue

        # 计算该大学的指标
        uni_indicators = calculate_university_indicators(connection, university["id"])

        # 计算相似度
        similarity = calculate_similarity(ecnu_indicators, uni_indicators)

        similarities.append(
            {
                "university_id": university["id"],
                "university_name": university["name"],
                "similarity": similarity,
                "indicators": uni_indicators,
            }
        )

    # 按相似度排序
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    return similarities[:top_n]


def main():
    """主函数"""
    connection = None
    try:
        # 连接数据库
        connection = connect_database()

        # 加载华东师范大学的指标
        ecnu_indicators = load_ecnu_indicators()

        print("华东师范大学关键指标:")
        print("-" * 50)
        for key, value in ecnu_indicators.items():
            if isinstance(value, list):
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value}")

        # 查找最相似的大学
        print("\n正在查找与华东师范大学最相似的大学...")
        similar_unis = find_similar_universities(connection, ecnu_indicators, top_n=10)

        print("\n与华东师范大学最相似的大学:")
        print("-" * 50)
        for i, uni in enumerate(similar_unis[:10], 1):
            print(f"{i}. {uni['university_name']} (相似度: {uni['similarity']:.4f})")
            # 显示该大学的前500名学科
            top_subjects = uni["indicators"].get("进入前500名的学科", [])
            if top_subjects:
                print(f"   进入前500名的学科: {', '.join(top_subjects)}")
            else:
                print("   无学科进入前500名")

        # 保存结果到CSV文件，包括各大学的详细指标
        output_dir = "QueryData"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, "similar_universities.csv")
        with open(filepath, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            headers = ["排名", "大学名称", "相似度"]
            indicator_names = list(ecnu_indicators.keys())
            # 移除列表类型的指标，因为它们无法很好地写入CSV
            numeric_indicator_names = [
                name for name in indicator_names if not isinstance(ecnu_indicators[name], list)
            ]
            headers.extend(numeric_indicator_names)
            headers.append("进入前500名的学科")  # 单独添加这个指标
            writer.writerow(headers)

            # 写入数据
            for i, uni in enumerate(similar_unis, 1):
                row = [i, uni["university_name"], f"{uni['similarity']:.4f}"]
                # 添加该大学的各项数值型指标数据
                for indicator_name in numeric_indicator_names:
                    indicator_value = uni["indicators"].get(indicator_name, 0)
                    row.append(indicator_value)
                # 单独处理列表类型的指标
                top_subjects = uni["indicators"].get("进入前500名的学科", [])
                row.append("; ".join(top_subjects))
                writer.writerow(row)
        print(f"\n结果已保存到 {filepath}")

    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
    finally:
        if connection:
            connection.close()
            print("数据库连接已关闭")


if __name__ == "__main__":
    main()
