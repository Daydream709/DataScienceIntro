import os
import csv
import pymysql
from typing import Dict

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "tyj810915mysql",
    "database": "esi_data",
    "charset": "utf8mb4",
}

# 学科名称映射（处理文件名到数据库学科名）
SUBJECT_MAPPING = {
    "AGRICULTURAL SCIENCES": "AGRICULTURAL SCIENCES",
    "BIOLOGY & BIOCHEMISTRY": "BIOLOGY & BIOCHEMISTRY",
    "CHEMISTRY": "CHEMISTRY",
    "CLINICAL MEDICINE": "CLINICAL MEDICINE",
    "COMPUTER SCIENCE": "COMPUTER SCIENCE",
    "ECONOMICS & BUSINESS": "ECONOMICS & BUSINESS",
    "ENGINEERING": "ENGINEERING",
    "ENVIRONMENT ECOLOGY": "ENVIRONMENT ECOLOGY",
    "GEOSCIENCES": "GEOSCIENCES",
    "IMMUNOLOGY": "IMMUNOLOGY",
    "MATERIALS SCIENCE": "MATERIALS SCIENCE",
    "MATHEMATICS": "MATHEMATICS",
    "MICROBIOLOGY": "MICROBIOLOGY",
    "MOLECULAR BIOLOGY & GENETICS": "MOLECULAR BIOLOGY & GENETICS",
    "MULTIDISCIPLINARY": "MULTIDISCIPLINARY",
    "NEUROSCIENCE & BEHAVIOR": "NEUROSCIENCE & BEHAVIOR",
    "PHARMACOLOGY & TOXICOLOGY": "PHARMACOLOGY & TOXICOLOGY",
    "PHYSICS": "PHYSICS",
    "PLANT & ANIMAL SCIENCE": "PLANT & ANIMAL SCIENCE",
    "PSYCHIATRY PSYCHOLOGY": "PSYCHIATRY PSYCHOLOGY",
    "SOCIAL SCIENCES, GENERAL": "SOCIAL SCIENCES, GENERAL",
    "SPACE SCIENCE": "SPACE SCIENCE",
}


class ESIDataImporter:
    def __init__(self):
        self.connection = None
        self.subject_ids = {}  # 缓存学科ID
        self.university_ids = {}  # 缓存大学ID
        self.country_ids = {}  # 缓存国家ID

    def connect_database(self):
        """连接数据库"""
        try:
            self.connection = pymysql.connect(**DB_CONFIG)
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise

    def close_connection(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("数据库连接已关闭")

    def insert_or_get_subject(self, subject_name: str) -> int:
        """插入或获取学科ID"""
        if subject_name in self.subject_ids:
            return self.subject_ids[subject_name]

        cursor = self.connection.cursor()
        try:
            # 检查学科是否已存在
            cursor.execute("SELECT id FROM subjects WHERE name = %s", (subject_name,))
            result = cursor.fetchone()

            if result:
                subject_id = result[0]
            else:
                # 插入新学科
                cursor.execute("INSERT INTO subjects (name) VALUES (%s)", (subject_name,))
                subject_id = cursor.lastrowid

            self.connection.commit()
            self.subject_ids[subject_name] = subject_id
            return subject_id
        finally:
            cursor.close()

    def insert_or_get_country(self, country_name: str) -> int:
        """插入或获取国家ID"""
        # 定义国家和区域的映射
        region_mapping = {
            "CHINA MAINLAND": "Asia",
            "USA": "North America",
            "ENGLAND": "Europe",
            "GERMANY (FED REP GER)": "Europe",
            "FRANCE": "Europe",
            "JAPAN": "Asia",
            "CANADA": "North America",
            "SOUTH KOREA": "Asia",
            "AUSTRALIA": "Oceania",
            "INDIA": "Asia",
            "SPAIN": "Europe",
            "ITALY": "Europe",
            "NETHERLANDS": "Europe",
            "SWITZERLAND": "Europe",
            "RUSSIA": "Europe",
            "ISRAEL": "Asia",
            "BRAZIL": "South America",
            "IRAN": "Asia",
            "POLAND": "Europe",
            "BELGIUM": "Europe",
            "TURKIYE": "Asia",
            "SOUTH AFRICA": "Africa",
            "MALAYSIA": "Asia",
            "FINLAND": "Europe",
            "SCOTLAND": "Europe",
            "AUSTRIA": "Europe",
            "IRELAND": "Europe",
            "NORWAY": "Europe",
            "SWEDEN": "Europe",
            "DENMARK": "Europe",
            "CZECH REPUBLIC": "Europe",
            "PORTUGAL": "Europe",
            "HUNGARY": "Europe",
            "EGYPT": "Africa",
            "ARGENTINA": "South America",
            "CHILE": "South America",
            "SAUDI ARABIA": "Asia",
            "NEW ZEALAND": "Oceania",
            "UKRAINE": "Europe",
            "GREECE": "Europe",
            "ROMANIA": "Europe",
            "PAKISTAN": "Asia",
            "SINGAPORE": "Asia",
            "SERBIA": "Europe",
            "CROATIA": "Europe",
            "MEXICO": "North America",
            "THAILAND": "Asia",
            "TUNISIA": "Africa",
            "TAIWAN": "Asia",
            "HONG KONG": "Asia",
            "PHILIPPINES": "Asia",
            "NORTHERN IRELAND": "Europe",
            "URUGUAY": "South America",
            "COLOMBIA": "South America",
            "SLOVENIA": "Europe",
            "KENYA": "Africa",
            "WALES": "Europe",
            "UNITED ARAB EMIRATES": "Asia",
            "MACAU": "Asia",
            "BANGLADESH": "Asia",
            "OMAN": "Asia",
            "NIGERIA": "Africa",
            "ETHIOPIA": "Africa",
            "GHANA": "Africa",
            "SLOVAKIA": "Europe",
            "COTE IVOIRE": "Africa",
            "ALGERIA": "Africa",
            "PERU": "South America",
            "VIETNAM": "Asia",
            "SRI LANKA": "Asia",
            "MOROCCO": "Africa",
            "LITHUANIA": "Europe",
            "INDONESIA": "Asia",
            "QATAR": "Asia",
            "JORDAN": "Asia",
            "LEBANON": "Asia",
            "CYPRUS": "Europe",
            "ZIMBABWE": "Africa",
            "BENIN": "Africa",
            "UGANDA": "Africa",
            "BULGARIA": "Europe",
            "ESTONIA": "Europe",
            "LUXEMBOURG": "Europe",
            "MALTA": "Europe",
            "ICELAND": "Europe",
            "PANAMA": "North America",
            "LATVIA": "Europe",
            "BELARUS": "Europe",
            "KAZAKHSTAN": "Asia",
            "BRUNEI": "Asia",
            "CUBA": "North America",
            "IRAQ": "Asia",
            "KUWAIT": "Asia",
            "CAMEROON": "Africa",
            "BAHRAIN": "Asia",
            "KYRGYZSTAN": "Asia",
            "PALESTINE": "Asia",
            "ZAMBIA": "Africa",
            "TANZANIA": "Africa",
            "MOZAMBIQUE": "Africa",
            "JAMAICA": "North America",
            "DEMOCRATIC REPUBLIC OF THE CONGO": "Africa",
            "COSTA RICA": "North America",
            "NEPAL": "Asia",
            "MALAWI": "Africa",
            "GEORGIA": "Asia",
            "LIBYA": "Africa",
            "BOTSWANA": "Africa",
            "VENEZUELA": "South America",
            "NAMIBIA": "Africa",
            "PAPUA NEW GUINEA": "Oceania",
            "BOSNIA & HERZEGOVINA": "Europe",
            "TRINIDAD TOBAGO": "North America",
            "ECUADOR": "South America",
            "ARMENIA": "Asia",
            "RWANDA": "Africa",
            "AZERBAIJAN": "Asia",
            "MONGOLIA": "Asia",
            "PARAGUAY": "South America",
            "MACEDONIA": "Europe",
            "BARBADOS": "North America",
            "GAMBIA": "Africa",
            "MOLDOVA": "Europe",
            "SUDAN": "Africa",
            "FIJI": "Oceania",
            "SENEGAL": "Africa",
            "MONTENEGRO": "Europe",
            "YEMEN": "Asia",
            "SURINAME": "South America",
            "CAMBODIA": "Asia",
            "MALI": "Africa",
            "KOSOVO": "Europe",
            "GUATEMALA": "North America",
            "GREENLAND": "North America",
            "BOLIVIA": "South America",
            "BERMUDA": "North America",
            "MAURITIUS": "Africa",
            "UZBEKISTAN": "Asia",
        }

        # 默认区域
        region = region_mapping.get(country_name, "其他")

        if country_name in self.country_ids:
            return self.country_ids[country_name]

        cursor = self.connection.cursor()
        try:
            # 检查国家是否已存在
            cursor.execute("SELECT id FROM countries WHERE name = %s", (country_name,))
            result = cursor.fetchone()

            if result:
                country_id = result[0]
            else:
                # 插入新国家
                cursor.execute("INSERT INTO countries (name, region) VALUES (%s, %s)", (country_name, region))
                country_id = cursor.lastrowid

            self.connection.commit()
            self.country_ids[country_name] = country_id
            return country_id
        finally:
            cursor.close()

    def insert_or_get_university(self, university_name: str, country_name: str) -> int:
        """插入或获取大学ID"""
        key = (university_name, country_name)
        if key in self.university_ids:
            return self.university_ids[key]

        cursor = self.connection.cursor()
        try:
            # 获取国家ID
            country_id = self.insert_or_get_country(country_name)

            # 检查大学是否已存在
            cursor.execute(
                """
                SELECT id FROM universities 
                WHERE name = %s AND country_id = %s
            """,
                (university_name, country_id),
            )
            result = cursor.fetchone()

            if result:
                university_id = result[0]
            else:
                # 插入新大学
                cursor.execute(
                    """
                    INSERT INTO universities (name, country_id) 
                    VALUES (%s, %s)
                """,
                    (university_name, country_id),
                )
                university_id = cursor.lastrowid

            self.connection.commit()
            self.university_ids[key] = university_id
            return university_id
        finally:
            cursor.close()

    def process_csv_file(self, file_path: str, subject_name: str):
        """处理单个CSV文件"""
        print(f"正在处理文件: {file_path}, 学科: {subject_name}")

        # 获取学科ID
        subject_id = self.insert_or_get_subject(subject_name)

        cursor = self.connection.cursor()
        try:
            # 使用ISO-8859-1编码读取文件
            with open(file_path, "r", encoding="ISO-8859-1") as csvfile:
                # 跳过第一行标题行
                csvfile.readline()  # Indicators Results List: Institutions...

                # 读取CSV数据
                reader = csv.reader(csvfile)
                next(reader)  # 跳过列标题行

                row_num = 0  # 初始化行号计数器
                for row in reader:
                    row_num += 1  # 行号从1开始计数
                    if len(row) >= 7:  # 确保有足够的列
                        try:
                            # 解析数据
                            rank_order = int(row[0].strip('"')) if row[0].strip('"') else 0
                            university_name = row[1].strip('"')
                            country_name = row[2].strip('"')
                            papers = int(row[3].strip('"')) if row[3].strip('"') else 0
                            citations = int(row[4].strip('"')) if row[4].strip('"') else 0
                            cites_per_paper = float(row[5].strip('"')) if row[5].strip('"') else 0.0
                            # 注意：CSV中第7列是top_papers_count，但我们数据库中没有这个字段

                            # 获取大学ID
                            university_id = self.insert_or_get_university(university_name, country_name)

                            # 插入排名数据
                            cursor.execute(
                                """
                                INSERT INTO rankings 
                                (university_id, subject_id, year, rank_order, 
                                 papers, citations, cites_per_paper)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON DUPLICATE KEY UPDATE
                                papers = VALUES(papers),
                                citations = VALUES(citations),
                                cites_per_paper = VALUES(cites_per_paper)
                            """,
                                (
                                    university_id,
                                    subject_id,
                                    2025,  # 默认年份
                                    rank_order,
                                    papers,
                                    citations,
                                    cites_per_paper,
                                ),
                            )

                        except ValueError as e:
                            print(f"解析第{row_num}行数据时出错: {row}, 错误: {e}")
                            continue
                        except Exception as e:
                            print(f"处理第{row_num}行数据时发生未知错误: {row}, 错误: {e}")
                            continue

            self.connection.commit()
            print(f"完成处理文件: {file_path}")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
        finally:
            cursor.close()

    def import_all_data(self, data_folder: str):
        """导入所有数据"""
        print(f"开始从文件夹导入数据: {data_folder}")

        # 遍历所有CSV文件
        for filename in os.listdir(data_folder):
            if filename.endswith(".csv"):
                # 从文件名提取学科名称
                subject_name = filename.replace(".csv", "")

                # 如果有映射则使用映射名称
                if subject_name in SUBJECT_MAPPING:
                    subject_name = SUBJECT_MAPPING[subject_name]

                file_path = os.path.join(data_folder, filename)
                try:
                    self.process_csv_file(file_path, subject_name)
                except Exception as e:
                    print(f"处理文件 {filename} 时发生错误: {e}")
                    continue

        print("所有数据导入完成")

    def run(self):
        """运行导入程序"""
        try:
            self.connect_database()
            data_folder = os.path.join(os.path.dirname(__file__), "download")
            self.import_all_data(data_folder)
        except Exception as e:
            print(f"导入过程中出现错误: {e}")
        finally:
            self.close_connection()


if __name__ == "__main__":
    importer = ESIDataImporter()
    importer.run()
