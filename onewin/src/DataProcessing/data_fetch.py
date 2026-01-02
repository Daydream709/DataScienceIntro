import csv
import os
import asyncio
import aiohttp
from lxml import html
import re
import random
from charset_normalizer import from_bytes


def extract_race_table_from_html(html_content, url):
    """
    从HTML内容中提取赛马结果表格数据

    Args:
        html_content (str): HTML内容
        url (str): 来源URL

    Returns:
        list: 提取的数据行，如果失败返回None
    """
    try:
        # 使用charset_normalizer检测编码并转换为字符串
        if isinstance(html_content, bytes):
            # 使用charset_normalizer检测编码
            detected = from_bytes(html_content)
            html_content = str(detected.best())
        elif not isinstance(html_content, str):
            html_content = str(html_content)

        tree = html.fromstring(html_content)

        # 检查比赛是否存在 - 通过检查smalltxt段落中是否包含"1970年01月01日"
        smalltxt_elements = tree.xpath("//p[@class='smalltxt']")
        if smalltxt_elements:
            smalltxt_content = smalltxt_elements[0].text_content()
            if "1970年01月01日" in smalltxt_content:
                print(f"比赛不存在: {url}")
                return None

        # 提取场地信息
        race_info_elements = tree.xpath("//span[contains(text(), '芝') or contains(text(), 'ダ')]")
        race_info_text = ""
        if race_info_elements:
            race_info_text = race_info_elements[0].text_content().strip()

        # 解析场地信息
        track_type = "芝" if "芝" in race_info_text else "ダ" if "ダ" in race_info_text else ""
        direction = "右" if "右" in race_info_text else "左" if "左" in race_info_text else ""

        # 提取距离
        distance_match = re.search(r"(\d+)m", race_info_text)
        distance = distance_match.group(1) if distance_match else ""

        # 提取天气
        weather_match = re.search(r"天候\s*:\s*([^/&nbsp;]+)", html_content)
        weather = weather_match.group(1).strip() if weather_match else ""

        # 提取场地情况
        track_condition_match = re.search(r"(?:芝|ダート)\s*:\s*([^/&nbsp;]+)", html_content)
        track_condition = track_condition_match.group(1).strip() if track_condition_match else ""

        # 使用XPath选择表格
        table_xpath = "//table[@class='race_table_01 nk_tb_common']"
        tables = tree.xpath(table_xpath)

        if not tables:
            print(f"在 {url} 中未找到目标表格")
            return None

        # 选择第一个表格
        table = tables[0]

        # 定义需要保留的列索引（从0开始）
        # 0: 着順, 1: 枠番, 3: 馬名, 4: 性齢, 5: 斤量, 6: 騎手, 7: タイム, 8: 着差, 11: 上り, 12: 単勝, 13: 人気, 14: 馬体重
        keep_columns = [0, 1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]

        # 定义新的表头（包含场地信息）
        new_headers = [
            "着順",
            "枠番",
            "馬名",
            "性齢",
            "斤量",
            "騎手",
            "タイム",
            "着差",
            "上り",
            "単勝",
            "人気",
            "馬体重",
            "場地类型",  # 新增列
            "方向",  # 新增列
            "距離",  # 新增列
            "天気",  # 新增列
            "場地状況",  # 新增列
        ]

        # 提取表格数据
        rows = []
        data_rows = table.xpath(".//tr[not(@class='txt_c')]")

        for row in data_rows:
            cells = row.xpath(".//td")
            row_data = []

            for i, cell in enumerate(cells):
                # 只处理需要保留的列
                if i in keep_columns:
                    cell_text = cell.text_content().strip()

                    # 特殊处理某些列，如提取链接中的马名、骑手名等
                    if i == 3:  # 马名列
                        # 提取马名链接中的文本
                        links = cell.xpath(".//a/text()")
                        if links:
                            cell_text = links[0].strip()

                    # 处理骑手名（第6列，索引为6）
                    elif i == 6:
                        links = cell.xpath(".//a/text()")
                        if links:
                            cell_text = links[0].strip()

                    row_data.append(cell_text)

            # 添加场地信息
            row_data.extend([track_type, direction, distance, weather, track_condition])

            if row_data:  # 只添加非空行
                rows.append(row_data)

        return {
            "headers": new_headers,
            "rows": rows,
            "race_id": url.split("/")[-2],  # 提取race_id，如202507050101
        }

    except Exception as e:
        print(f"解析HTML时出错: {e}")
        return None


def save_to_csv(data, output_path):
    """
    将数据保存到CSV文件

    Args:
        data (dict): 包含headers和rows的数据字典
        output_path (str): 输出文件路径
    """
    if data is None or not data["rows"]:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(data["headers"])

        # 写入数据行
        for row in data["rows"]:
            writer.writerow(row)


# 创建一个信号量来限制并发请求数
semaphore = asyncio.Semaphore(14)  # 限制同时最多5个请求


async def fetch_race_data(session, url, year_folder):
    """
    异步获取单个比赛的数据
    """
    async with semaphore:  # 限制并发数
        try:
            await asyncio.sleep(random.randint(15, 20) / 10)  # 请求前延迟，避免过于频繁
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"        HTTP错误 {response.status}，跳过: {url}")
                    return False

                # 获取响应内容
                html_content = await response.read()
                
                # 使用charset_normalizer检测编码
                detected = from_bytes(html_content)
                html_text = str(detected.best())

                # 尝试解析HTML
                data = extract_race_table_from_html(html_text, url)

                if data and data["rows"]:
                    # 保存数据
                    race_id = url.split("/")[-2]  # 提取race_id，如202507050101
                    output_path = os.path.join(year_folder, f"{race_id}.csv")
                    save_to_csv(data, output_path)
                    print(f"        成功保存: {output_path}")
                    return True
                else:
                    # 如果数据为空或比赛不存在，说明当前比赛编号已结束
                    print(f"        比赛不存在或无数据，结束当前比赛循环: {url}")
                    return False

        except Exception as e:
            print(f"        请求失败: {e}，跳过: {url}")
            return False


async def generate_race_urls(base_url, year, course, race_day, day):
    """
    生成比赛URL列表
    """
    urls = []
    race = 1
    while True:
        race_str = f"{race:02d}"
        race_id = f"{year}{course:02d}{race_day:02d}{day:02d}{race_str}"
        url = f"{base_url}{race_id}/"
        urls.append((race_id, url))
        race += 1
        
        # 限制单日比赛数量，避免无限循环
        if race > 20:  # 假设每天最多20场比赛
            break
            
    return urls


async def process_day_races(session, base_url, year, course, race_day, day, year_folder):
    """
    异步处理单日的所有比赛
    """
    # 生成该天所有比赛的URL
    urls = await generate_race_urls(base_url, year, course, race_day, day)

    # 创建所有请求任务
    tasks = [fetch_race_data(session, url, year_folder) for _, url in urls]

    # 执行所有请求
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 统计成功的请求数量
    success_count = 0
    for r in results:
        if not isinstance(r, Exception) and r is True:
            success_count += 1

    return success_count > 0


async def process_race_day(session, base_url, year, course, race_day, year_folder):
    """
    异步处理一个比赛日的所有比赛
    """
    race_day_str = f"{race_day:02d}"
    print(f"    处理第{race_day_str}回比赛")

    day = 1
    race_day_success = False

    while True:
        day_str = f"{day:02d}"
        print(f"      处理第{day_str}日比赛")

        # 处理当天所有比赛
        day_success = await process_day_races(session, base_url, year, course, race_day, day, year_folder)

        if not day_success:
            break  # 当天没有成功获取数据，说明比赛日结束了
        race_day_success = True
        day += 1
        
        # 限制天数，避免无限循环
        if day > 10:  # 假设每个比赛日最多10天
            break

    return race_day_success


async def process_course_async(base_url, year, course, year_folder):
    """
    异步处理一个马场的所有比赛
    """
    print(f"  处理马场: {course:02d}")

    async with aiohttp.ClientSession(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        race_day = 1
        while True:
            race_day_str = f"{race_day:02d}"
            print(f"    处理第{race_day_str}回比赛")

            # 检查这个比赛日是否有数据
            race_day_success = await process_race_day(session, base_url, year, course, race_day, year_folder)

            if not race_day_success:
                break  # 这个比赛日没有成功获取数据，说明比赛大会结束了
            race_day += 1
            
            # 限制比赛日数量，避免无限循环
            if race_day > 15:  # 假设每个马场最多15个比赛日
                break


def crawl_race_data():
    """
    爬取所有年份、马场、比赛日和比赛的数据
    """
    base_url = "https://db.netkeiba.com/race/"

    # 创建data文件夹
    os.makedirs("data", exist_ok=True)

    # 年份循环：2020-2025
    for year in range(2019, 2020):
        print(f"开始处理年份: {year}")

        # 创建年份文件夹
        year_folder = os.path.join("data", str(year))
        os.makedirs(year_folder, exist_ok=True)

        # 马场循环：01-10
        for course in range(1, 11):
            # 同步处理每个马场，以避免过于频繁的请求
            asyncio.run(process_course_async(base_url, year, course, year_folder))

        print(f"完成年份: {year}")


def main():
    print("开始爬取赛马数据...")
    crawl_race_data()
    print("爬取完成！")


if __name__ == "__main__":
    main()
