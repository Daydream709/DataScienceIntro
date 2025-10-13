import requests
import json

filtervalues = ""
print(
    "1.Agricultural Sciences\n2.Biology & Biochemistry\n3.Chemistry\n4.Clinical Medicine\n5.Computer Science\n6.Economics & Business\n7.Engineering\n8.Environment/Ecology\n9.Geosciences\n10.Immunology\n11.Materials Science\n12.Mathematics\n13.Microbiology\n14.Molecular Biology & Genetics\n15.Multidisciplinary\n16.Neuroscience & Behavior\n17.Pharmacology & Toxicology\n18.Physics\n19.Plant & Animal Science\n20.Psychiatry/Psychology\n21.Social Sciences, General\n22.Space Science"
)
researchfields = {
    1: "AGRICULTURAL SCIENCES",
    2: "BIOLOGY & BIOCHEMISTRY",
    3: "CHEMISTRY",
    4: "CLINICAL MEDICINE",
    5: "COMPUTER SCIENCE",
    6: "ECONOMICS & BUSINESS",
    7: "ENGINEERING",
    8: "ENVIRONMENT/ECOLOGY",
    9: "GEOSCIENCES",
    10: "IMMUNOLOGY",
    11: "MATERIALS SCIENCE",
    12: "MATHEMATICS",
    13: "MICROBIOLOGY",
    14: "MOLECULAR BIOLOGY & GENETICS",
    15: "MULTIDISCIPLINARY",
    16: "NEUROSCIENCE & BEHAVIOR",
    17: "PHARMACOLOGY & TOXICOLOGY",
    18: "PHYSICS",
    19: "PLANT & ANIMAL SCIENCE",
    20: "PSYCHIATRY/PSYCHOLOGY",
    21: "SOCIAL SCIENCES, GENERAL",
    22: "SPACE SCIENCE",
}

choice = int(input("请输入您要查询的排名数："))
# 构建URL和参数
url = "https://esi.clarivate.com/IndicatorsDataAction.action"
params = {
    "_dc": "1759388434752",
    "type": "grid",
    "groupBy": "Institutions",
    "filterBy": "ResearchFields",
    "filterValues": researchfields[choice],
    "docType": "Top",
    "page": "1",
    "start": "0",
    "limit": "10000",
    "sort": json.dumps([{"property": "cites", "direction": "DESC"}]),
}

# 设置请求头
headers = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "dnt": "1",
    "prefer": "safe",
    "priority": "u=1, i",
    "referer": "https://esi.clarivate.com/IndicatorsAction.action?app=esi&Init=Yes&authCode=null&SrcApp=IC2LS&SID=H3-er5qhfxxbvx2BxxIXIgCMfIBHr90EKTkeFSh-18x2dAV2x2BRfMwnYLsoix2BefrHXeAx3Dx3Dsg4JL3RQx2FksWHsK5x2Fo1rKQx3Dx3D-deDoSViHIQYUGXyhfV4d4Ax3Dx3D-ucx2FlMPFCLJrFFs0K4gTuzQx3Dx3D",
    "sec-ch-ua": '"Chromium";v="140", "Not=A?Brand";v="24", "Microsoft Edge";v="140"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0",
    "x-requested-with": "XMLHttpRequest",
}

# 设置cookies
cookies = {
    "_vwo_uuid_v2": "DEAEF3E85609D025FAF5A7271E2A88042|019da3240b98d291951e1f85ac059fee",
    "_vwo_uuid": "DEAEF3E85609D025FAF5A7271E2A88042",
    "_vis_opt_s": "2|",
    "_vis_opt_test_cookie": "1",
    "ELOQUA": "GUID=3A1A48E729B14C54B5B7ACBB11FC3287",
    "_biz_uid": "ce0ca1da8b94429ec932759ae7ae15b6",
    "_clck": "5es48d^2^fzt^0^2098",
    "_biz_flagsA": '{"Version":1,"ViewThrough":"1","XDomain":"1"}',
    "_gcl_au": "1.1.560584855.1759140469",
    "OptanonAlertBoxClosed": "2025-09-29T10:07:53.314Z",
    "_vwo_consent": "1%2C1%3A~",
    "_fbp": "fb.1.1759140538280.220285229362047850",
    "_zitok": "72ba72de22fea087d7be1759140542",
    "_gid": "GA1.2.385096944.1759388153",
    "PSSID": '"H3-er5qhfxxbvx2BxxIXIgCMfIBHr90EKTkeFSh-18x2dAV2x2BRfMwnYLsoix2BefrHXeAx3Dx3Dsg4JL3RQx2FksWHsK5x2Fo1rKQx3Dx3D-deDoSViHIQYUGXyhfV4d4Ax3Dx3D-ucx2FlMPFCLJrFFs0K4gTuzQx3Dx3D"',
    "IC2_SID": '"H3-er5qhfxxbvx2BxxIXIgCMfIBHr90EKTkeFSh-18x2dAV2x2BRfMwnYLsoix2BefrHXeAx3Dx3Dsg4JL3RQx2FksWHsK5x2Fo1rKQx3Dx3D-deDoSViHIQYUGXyhfV4d4Ax3Dx3D-ucx2FlMPFCLJrFFs0K4gTuzQx3Dx3D"',
    "CUSTOMER_NAME": '"EAST CHINA NORMAL UNIV"',
    "E_GROUP_NAME": '"IC2 Platform"',
    "SUBSCRIPTION_GROUP_ID": '"260055"',
    "SUBSCRIPTION_GROUP_NAME": '"EAST CHINA NORMAL UNIV_20151126590_1"',
    "CUSTOMER_GROUP_ID": '"99582"',
    "IP_SET_ID_NAME": '"E China Normal U"',
    "IP_SET_ID": '"3204746"',
    "ROAMING_DISABLED": '"true"',
    "ACCESS_METHOD": '"IP"',
    "userAuthType": '"TrustedIPAuth"',
    "userAuthIDType": '"222.66.117.97"',
    "esi.isLocalStorageCleared": "true",
    "_sp_ses.2f26": "*",
    "esi.Show": "",
    "esi.Type": "",
    "esi.FilterValue": "",
    "esi.GroupBy": "",
    "esi.FilterBy": "",
    "esi.authorsList": "",
    "esi.frontList": "",
    "esi.fieldsList": "",
    "esi.instList": "",
    "esi.journalList": "",
    "esi.terriList": "",
    "esi.titleList": "",
    "OptanonConsent": "isGpcEnabled=0&datestamp=Thu+Oct+02+2025+14%3A58%3A04+GMT%2B0800+(%E4%B8%AD%E5%9B%BD%E6%A0%87%E5%87%86%E6%97%B6%E9%97%B4)&version=202503.2.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=36de2f65-742b-4d70-92f3-65952260373b&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0003%3A1%2CC0004%3A1%2CC0002%3A1&intType=1&geolocation=CN%3BSH&AwaitingReconsent=false",
    "_biz_nA": "6",
    "_vwo_ds": "3%3At_1%2Ca_1%3A0%241759140459%3A41.62913555%3A393_1_0_2%2C308_0_1_0_1%2C311_0_1_0_1%2C312_0_1_0_1%3A%3A298_1%2C297_1%2C3_1%2C2_1%3A16",
    "_biz_pendingA": "[]",
    "_uetsid": "24f355d09f5d11f082c049d1e7158f4e|18yznmk|2|fzt|0|2101",
    "_vwo_sn": "247789%3A1%3Ar3.visualwebsiteoptimizer.com%3A1%3A1%3Areferrer%3D",
    "_clsk": "1xjk99s^1759388288212^1^1^o.clarity.ms/collect",
    "_uetvid": "26b96bf09d1c11f0b7f5fb73dbf37a49|1xb83rc|1759388285965|1|1|bat.bing.com/p/insights/c/o",
    "_ga_9R70GJ8HZF": "GS2.1.s1759388285$o2$g1$t1759388335$j10$l0$h785171601",
    "_ga_K6K0YXL6HJ": "GS2.1.s1759388285$o2$g1$t1759388335$j10$l0$h965118851",
    "_ga_V1YLG54MGT": "GS2.1.s1759388285$o2$g1$t1759388335$j10$l0$h1986700553",
    "_ga": "GA1.2.330277511.1759140474",
    "JSESSIONID": "35A25425A75257A358F319EA1C9BE393",
    "__cf_bm": "utXgybiiFHWyuZNjaD8DiFwFdI78p5Elu6Pu7Tmg0Dw-1759388424-1.0.1.1-KfsK2DqGKApJcAvaYvD9qsGBrdR5ONW2kEE5UXPoFObQDrrrY8kxMtHGy0Er.v2yex5JjDlL2pCGGzxNbnT0uoYxDJlDKsOZPNOm9TgxVcc",
    "_gat": "1",
    "_ga_D5KRF08D0Q": "GS2.2.s1759388373$o2$g1$t1759388434$j60$l0$h0",
    "_sp_id.2f26": "2dcf8224-a996-46a0-8e61-cb92864ac9b3.1759140689.2.1759388435.1759142760.ffd7b905-ed80-4877-9c7c-bd3a047bd365",
}


response = requests.get(url, params=params, headers=headers, cookies=cookies)
# 打印状态码和响应头
print(response.status_code)
print(response.headers)
try:
    data = response.json()
    print("解析成功:")
    filename = researchfields[choice].replace("/", "_") + ".json"
    # 将数据保存到JSON文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("数据已保存到" + filename + "文件中")

except json.JSONDecodeError as e:
    print("解析失败:", e)
