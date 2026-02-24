#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成拟人化的训练数据
"""

import pandas as pd
import random
import json

# 设置随机种子以保证可复现
random.seed(42)

# 存储所有训练数据
training_data = []

# ==================== 1. call_phone ====================
def generate_call_phone():
    """拨打电话给某人"""
    prompts = []
    
    # 常见联系人
    contacts = [
        ("爸妈", ["13800138000", "13900139000"]),
        ("妈妈", ["13800138001"]),
        ("爸爸", ["13800138002"]),
        ("老婆", ["13800138003"]),
        ("老公", ["13800138004"]),
        ("女朋友", ["13800138005"]),
        ("男朋友", ["13800138006"]),
        ("老婆婆", ["13800138007"]),
        ("老公公", ["13800138008"]),
        ("奶奶", ["13800138009"]),
        ("爷爷", ["13800138010"]),
        ("外公", ["13800138011"]),
        ("外婆", ["13800138012"]),
        ("老板", ["13800138013"]),
        ("同事", ["13800138014"]),
        ("领导", ["13800138015"]),
        ("客户", ["13800138016"]),
        ("朋友", ["13800138017"]),
        ("老王", ["13800138018"]),
        ("李明", ["13800138019"]),
    ]
    
    # 号码库
    numbers = [
        "13800138000", "13900139000", "18600186000", "13500135000",
        "18800188000", "15800158000", "15100151000", "13000130000",
        "13100131000", "13200132000", "13300133000", "15000150000",
        "15200152000", "15300153000", "15600156000", "16600166000",
        "17500175000", "17600176000", "17800178000", "18000180000"
    ]
    
    # 模板 - 直接说名字
    templates_name = [
        "给{}打个电话",
        "帮我拨打{}的电话",
        "打电话给{}",
        "我要给{}打个电话",
        "帮我叫{}接电话",
        "呼叫{}",
        "联系一下{}",
        "给{}拨个号",
        "帮我打{}的电话",
        "我要找{}",
    ]
    
    # 模板 - 说号码
    templates_number = [
        "帮我拨打{}",
        "打个电话到这个号码{}",
        "拨打{}这个电话",
        "帮我打{}",
        "呼叫这个号码{}",
        "给这个号码打个电话{}",
        "拨{}",
        "打这个电话{}",
    ]
    
    # 生成基于名字的样本
    for contact, phone_numbers in contacts:
        for _ in range(5):
            template = random.choice(templates_name)
            prompts.append((template.format(contact), "call_phone", json.dumps({"callee": random.choice(phone_numbers)}, ensure_ascii=False)))
    
    # 生成基于号码的样本
    for number in numbers:
        for _ in range(5):
            template = random.choice(templates_number)
            prompts.append((template.format(number), "call_phone", json.dumps({"callee": number}, ensure_ascii=False)))
    
    # 更多自然对话
    natural_conversations = [
        "帮我打个电话告诉我老婆我今晚加班",
        "给我妈打个电话问问她身体怎么样",
        "拨打一下这个号码13800138000",
        "帮我叫外卖，先打个电话给商家",
        "给我爸打个电话，他手机好像打不通",
        "帮我拨打客服热线",
        "给快递员打个电话问问在哪",
        "帮我打110报警",
        "给120打电话叫救护车",
        "帮我拨打10086",
        "给物业打个电话",
        "联系不上老板了，帮他打个电话",
        "帮我打一下这个联系人",
        "呼叫联系人里的老婆",
        "拨打通讯录里妈妈的号码",
    ]
    
    for conv in natural_conversations:
        # 提取号码或使用默认
        import re
        match = re.search(r'1[3-9]\d{9}', conv)
        if match:
            phone = match.group()
        else:
            phone = random.choice(numbers)
        prompts.append((conv, "call_phone", json.dumps({"callee": phone}, ensure_ascii=False)))
    
    return prompts

# ==================== 2. get_time ====================
def generate_get_time():
    """查询当前时间"""
    prompts = []
    
    templates = [
        "现在几点了",
        "现在几点",
        "几点钟了",
        "现在几点几点了",
        "帮我看一下现在几点",
        "查一下现在的时间",
        "现在是什么时间",
        "几点了",
        "告诉我现在几点",
        "现在多少点了",
        "看一下现在几点",
        "查询当前时间",
        "帮我看看时间",
        "现在几点了呀",
        "几点了现在",
        "帮我查下时间",
        "当前时间是多少",
        "现在几点了麻烦告诉我",
        "能告诉我现在几点吗",
        "看看现在几点",
    ]
    
    # 不同时区
    timezones = ["Asia/Shanghai", "America/New_York", "Europe/London", "Asia/Tokyo", "America/Los_Angeles"]
    
    timezone_templates = [
        "现在纽约几点了",
        "伦敦现在几点",
        "东京现在是几点",
        "帮我查一下纽约现在几点",
        "洛杉矶现在几点啦",
        "日本现在几点",
        "美国现在几点",
        "英国现在几点",
        "查下纽约时间",
        "悉尼现在几点",
    ]
    
    for template in templates * 5:
        prompts.append((template, "get_time", json.dumps({}, ensure_ascii=False)))
    
    for template in timezone_templates:
        tz = "America/New_York" if "纽约" in template or "美国" in template else \
             "Europe/London" if "伦敦" in template or "英国" in template else \
             "Asia/Tokyo" if "东京" in template or "日本" in template else \
             "America/Los_Angeles" if "洛杉矶" in template else "Australia/Sydney"
        prompts.append((template, "get_time", json.dumps({"timezone": tz}, ensure_ascii=False)))
    
    return prompts

# ==================== 3. get_current_temperature ====================
def generate_get_current_temperature():
    """查询某地的温度"""
    prompts = []
    
    locations = [
        "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都", "重庆", "西安",
        "天津", "苏州", "郑州", "长沙", "沈阳", "青岛", "济南", "大连", "哈尔滨", "长春",
        "厦门", "福州", "昆明", "兰州", "石家庄", "太原", "合肥", "南昌", "南宁", "贵阳",
        "乌鲁木齐", "拉萨", "西宁", "银川", "呼和浩特", "香港", "澳门", "台北",
        "东京", "首尔", "纽约", "伦敦", "巴黎", "悉尼", "墨尔本", "新加坡", "曼谷", "普吉岛"
    ]
    
    templates = [
        "{}现在多少度",
        "{}今天几度",
        "{}的温度是多少",
        "{}现在多少度",
        "帮我查一下{}的温度",
        "{}今天天气怎么样",
        "{}冷不冷",
        "{}热不热",
        "{}的温度",
        "查询{}的温度",
        "{}今天多少度",
        "看看{}的温度",
        "{}现在气温多少",
        "{}的气温",
        "帮我看看{}多少度",
        "{}现在冷吗",
        "{}现在热吗",
        "{}今天温度怎么样",
        "{}的温度查询",
        "查下{}温度",
    ]
    
    for location in locations:
        for _ in range(2):
            template = random.choice(templates)
            prompts.append((template.format(location), "get_current_temperature", json.dumps({"location": location}, ensure_ascii=False)))
    
    return prompts

# ==================== 4. get_lunar_calendar ====================
def generate_get_lunar_calendar():
    """查询农历日期信息"""
    prompts = []
    
    templates = [
        "今天是农历几号",
        "查一下农历",
        "今天是什么日子",
        "今天的农历日期",
        "帮我看一下农历",
        "农历今天是多少",
        "查农历",
        "今天农历几号",
        "现在农历是",
        "看看今天农历",
        "农历日期查询",
        "今天的农历",
        "帮我查一下农历",
        "今天是农历多少",
        "农历信息",
        "今天对应的农历",
        "查下今天的农历",
        "农历日期是什么",
        "今天农历是什么",
        "看看今天是什么农历",
    ]
    
    formats = ["json", "txt"]
    
    for template in templates * 5:
        fmt = random.choice(formats)
        prompts.append((template, "get_lunar_calendar", json.dumps({"type": fmt}, ensure_ascii=False)))
    
    return prompts

# ==================== 5. get_phone_location ====================
def generate_get_phone_location():
    """查询手机号的归属地"""
    prompts = []
    
    numbers = [
        "13800138000", "13900139000", "18600186000", "13500135000", "18800188000",
        "15800158000", "15100151000", "13000130000", "13100131000", "13200132000",
        "13300133000", "15000150000", "15200152000", "15300153000", "15600156000",
        "16600166000", "17500175000", "17600176000", "17800178000", "18000180000",
        "18100181000", "18200182000", "18300183000", "18400184000", "19300193000",
        "19700197000", "19800198000", "19900199000", "17000170000", "17100171000"
    ]
    
    templates = [
        "查一下{}是哪里的",
        "{}这个号码是哪的",
        "{}归属地",
        "这个号码{}是哪的",
        "帮我查{}的归属地",
        "{}是哪的号码",
        "{}是哪个地区的",
        "查号码{}的归属地",
        "{}这个手机号是哪的",
        "看看{}是哪里",
        "查询{}的位置",
        "{}这个号是哪的",
        "{}是哪的号",
        "帮我看看{}是哪的",
        "{}号码归属地",
        "查下{}归属地",
        "{}是哪里的号码",
        "{}这个电话是哪的",
        "看看这个号{}是哪的",
        "帮我查一下{}归属地",
    ]
    
    for number in numbers:
        for _ in range(3):
            template = random.choice(templates)
            prompts.append((template.format(number), "get_phone_location", json.dumps({"tel": number}, ensure_ascii=False)))
    
    # 特殊场景
    special_templates = [
        "这个号码是哪里的",
        "帮我查一下这个号码",
        "看看这个手机号是哪的",
        "这个号是哪的",
    ]
    for template in special_templates:
        prompts.append((template, "get_phone_location", json.dumps({"tel": random.choice(numbers)}, ensure_ascii=False)))
    
    return prompts

# ==================== 6. get_simp_diary ====================
def generate_get_simp_diary():
    """生成一条舔狗日记"""
    prompts = []
    
    templates = [
        "给我整一个舔狗日记",
        "生成一个舔狗日记",
        "来条舔狗日记",
        "舔狗日记",
        "搞笑的舔狗日记",
        "来一个舔狗日记逗逗我",
        "生成舔狗日记",
        "给我看一个舔狗日记",
        "来段舔狗日记",
        "舔狗日记来一个",
        "来篇舔狗日记",
        "生成一段舔狗日记",
        "给我写个舔狗日记",
        "来点舔狗日记",
        "搞笑舔狗日记",
        "来一条舔狗日记",
        "帮我生成舔狗日记",
        "来一个有趣的舔狗日记",
        "看看舔狗日记",
        "来段搞笑的舔狗日记",
    ]
    
    formats = ["text", "json", "js"]
    
    for template in templates * 5:
        fmt = random.choice(formats)
        prompts.append((template, "get_simp_diary", json.dumps({"type": fmt}, ensure_ascii=False)))
    
    return prompts

# ==================== 7. get_gas_price ====================
def generate_get_gas_price():
    """查询某地油价"""
    prompts = []
    
    provinces = [
        "北京", "上海", "广东", "浙江", "江苏", "山东", "四川", "湖北", "湖南", "河南",
        "河北", "陕西", "安徽", "福建", "江西", "辽宁", "吉林", "黑龙江", "山西", "云南",
        "贵州", "甘肃", "新疆", "内蒙古", "宁夏", "青海", "西藏", "海南", "广西", "天津"
    ]
    
    templates = [
        "{}油价多少",
        "{}汽油价格",
        "{}今天油价",
        "查一下{}的油价",
        "{}92号汽油",
        "{}95号油价",
        "{}98号汽油",
        "{}柴油价格",
        "{}油价查询",
        "{}今日油价",
        "{}加油多少钱",
        "看看{}油价",
        "{}油价怎么样",
        "查询{}油价",
        "{}油价是多少钱",
        "{}油价现在多少",
        "{}汽油多少钱",
        "{}油价信息",
        "查下{}油价",
        "{}今日油价多少",
    ]
    
    formats = ["json", "txt"]
    
    for province in provinces:
        for _ in range(3):
            template = random.choice(templates)
            fmt = random.choice(formats)
            prompts.append((template.format(province), "get_gas_price", json.dumps({"province": province, "type": fmt}, ensure_ascii=False)))
    
    return prompts

# ==================== 8. get_express_info ====================
def generate_get_express_info():
    """物流单查询"""
    prompts = []
    
    # 常见快递单号格式
    express_ids = [
        "SF1234567890", "YT1234567890", "ZTO1234567890", "STO1234567890", "EMS1234567890",
        "JDVB123456789", "JDJX123456789", "UFO1234567890", "ZJS1234567890", "TNT1234567890",
        "DHL1234567890", "UPS1234567890", "FEDEX123456789", "AA1234567890", "EMS1234567",
        "8881234567890", "9988776655443", "123456789012345678", "ABC123456789", "DEF123456789",
    ]
    
    templates = [
        "查一下快递{}",
        "物流{}到哪了",
        "帮我查一下{}",
        "快递{}怎么还没到",
        "{}物流信息",
        "查物流{}",
        "看看{}到哪了",
        "{}快递到哪了",
        "帮我查{}物流",
        "{}到哪里了",
        "物流查询{}",
        "快递{}的物流",
        "查一下{}的快递",
        "{}物流怎么样",
        "看看快递{}",
        "查快递{}",
        "{}到哪了",
        "帮我查下{}",
        "{}快递信息",
        "物流{}查询",
    ]
    
    formats = ["json", "text"]
    
    for express_id in express_ids:
        for _ in range(5):
            template = random.choice(templates)
            fmt = random.choice(formats)
            prompts.append((template.format(express_id), "get_express_info", json.dumps({"id": express_id, "type": fmt}, ensure_ascii=False)))
    
    return prompts

# ==================== 9. get_box_office ====================
def generate_get_box_office():
    """电影票房查询"""
    prompts = []
    
    templates = [
        "查一下电影票房",
        "现在票房第一是谁",
        "看看票房排行",
        "今日票房",
        "票房排行",
        "电影票房排行",
        "查票房",
        "现在什么电影最火",
        "票房排名",
        "最近票房怎么样",
        "看看最近票房",
        "今日票房排行",
        "票房排行榜",
        "查一下最近票房",
        "现在票房最高",
        "票房查询",
        "哪些电影票房高",
        "今天票房怎么样",
        "电影票房情况",
        "票房信息",
    ]
    
    formats = ["json", "txt"]
    
    for template in templates * 5:
        fmt = random.choice(formats)
        prompts.append((template, "get_box_office", json.dumps({"type": fmt}, ensure_ascii=False)))
    
    return prompts

# ==================== 10. get_license_plate_value ====================
def generate_get_license_plate_value():
    """车牌估价"""
    prompts = []
    
    # 各地车牌
    plates = [
        "京A12345", "沪A12345", "粤B12345", "浙C12345", "苏D12345",
        "京A88888", "沪B66666", "粤C99999", "浙D55555", "苏E44444",
        "京N12345", "沪C88888", "粤A66666", "浙B77777", "苏A55555",
        "川A12345", "陕A12345", "鄂A12345", "湘A12345", "豫A12345",
        "鲁A12345", "辽A12345", "黑A12345", "吉A12345", "闽A12345",
        "皖A12345", "赣A12345", "桂A12345", "琼A12345", "贵A12345",
        "云A12345", "藏A12345", "青A12345", "宁A12345", "新A12345",
        "津A12345", "渝A12345", "冀A12345", "晋A12345", "蒙A12345",
    ]
    
    templates = [
        "这个车牌{}值多少钱",
        "帮我估一下{}车牌",
        "{}车牌价格",
        "{}这个车牌怎么样",
        "查一下{}车牌价值",
        "{}车牌值不值钱",
        "{}这个号牌",
        "帮我看看{}车牌",
        "{}牌照价格",
        "车牌{}估价",
        "查{}车牌",
        "{}车牌的价钱",
        "看一下{}车牌",
        "{}号牌值多少",
        "{}牌照值多少钱",
        "帮我查{}车牌",
        "{}这个车牌好吗",
        "车牌号{}价格",
        "查下{}值多少钱",
        "{}这个车牌值多少钱",
    ]
    
    for plate in plates:
        for _ in range(3):
            template = random.choice(templates)
            prompts.append((template.format(plate), "get_license_plate_value", json.dumps({"che": plate}, ensure_ascii=False)))
    
    return prompts

# ==================== 11. query_food_calories ====================
def generate_query_food_calories():
    """查询食物卡路里"""
    prompts = []
    
    foods = [
        "米饭", "馒头", "面条", "面包", "鸡蛋", "牛奶", "苹果", "香蕉", "橙子", "西瓜",
        "草莓", "葡萄", "桃子", "梨", "火龙果", "榴莲", "芒果", "菠萝", "黄瓜", "西红柿",
        "菠菜", "白菜", "土豆", "红薯", "玉米", "胡萝卜", "西兰花", "青椒", "茄子", "豆腐",
        "鸡胸肉", "牛肉", "猪肉", "鱼肉", "虾", "螃蟹", "牛奶", "酸奶", "可乐", "奶茶",
        "咖啡", "啤酒", "白酒", "红酒", "巧克力", "冰淇淋", "蛋糕", "饼干", "薯片", "方便面",
        "披萨", "汉堡", "炸鸡", "薯条", "沙拉", "炒饭", "炒面", "盖浇饭", "麻辣烫", "火锅",
        "红烧肉", "糖醋排骨", "宫保鸡丁", "麻婆豆腐", "鱼香肉丝", "青椒肉丝", "西红柿炒蛋",
    ]
    
    templates = [
        "{}有多少卡路里",
        "{}的热量",
        "{}卡路里",
        "查一下{}的热量",
        "{}多少卡",
        "{}的热量是多少",
        "{}吃了会胖吗",
        "{}有多少热量",
        "看看{}的热量",
        "{}的能量",
        "{}的卡路里",
        "帮我查{}热量",
        "{}高热量吗",
        "{}脂肪含量",
        "{}营养成分",
        "{}吃了会不会胖",
        "{}食物热量",
        "查询{}卡路里",
        "{}有多少大卡",
        "看看{}多少卡",
    ]
    
    for food in foods:
        for _ in range(2):
            template = random.choice(templates)
            prompts.append((template.format(food), "query_food_calories", json.dumps({"food": food}, ensure_ascii=False)))
    
    return prompts

# ==================== 12. get_website_favicon ====================
def generate_get_website_favicon():
    """获取网站图标"""
    prompts = []
    
    websites = [
        "baidu.com", "google.com", "taobao.com", "jd.com", "alipay.com",
        "qq.com", "weibo.com", "zhihu.com", "bilibili.com", "douyin.com",
        "toutiao.com", "amazon.com", "facebook.com", "twitter.com", "instagram.com",
        "youtube.com", "reddit.com", "wikipedia.org", "github.com", "stackoverflow.com",
        "microsoft.com", "apple.com", "huawei.com", "xiaomi.com", "oppo.com",
        "vivo.com", "meituan.com", "dianping.com", "ele.me", "ctrip.com",
    ]
    
    templates = [
        "帮我获取{}的图标",
        "{}的favicon",
        "{}网站图标",
        "拿一下{}的logo",
        "{}的icon",
        "获取{}网站图标",
        "{}网站favicon",
        "看看{}的图标",
        "{}网站logo",
        "帮我找{}的favicon",
        "{}网站图标在哪",
        "下载{}的图标",
        "{}的网站图标",
        "拿{}的favicon",
        "获取网站{}的图标",
        "{}网站favicon地址",
        "{}网站小图标",
        "{}网站图标链接",
        "帮我查{}的favicon",
        "{}的网站favicon",
    ]
    
    for website in websites:
        for _ in range(3):
            template = random.choice(templates)
            prompts.append((template.format(website), "get_website_favicon", json.dumps({"url": website}, ensure_ascii=False)))
    
    return prompts

# ==================== 13. get_webpage_images ====================
def generate_get_webpage_images():
    """抓取网站的图片"""
    prompts = []
    
    urls = [
        "https://www.baidu.com", "https://www.google.com", "https://www.taobao.com",
        "https://www.jd.com", "https://www.bilibili.com", "https://www.zhihu.com",
        "https://www.douyin.com", "https://www.weibo.com", "https://github.com",
        "https://stackoverflow.com", "https://www.youtube.com", "https://www.facebook.com",
        "https://twitter.com", "https://www.instagram.com", "https://www.amazon.com",
    ]
    
    templates = [
        "帮我抓取{}的图片",
        "{}有哪些图片",
        "获取{}页面图片",
        "{}网站的图片",
        "爬取{}的图片",
        "看看{}有什么图片",
        "{}页面上的图片",
        "提取{}的图片",
        "{}网页图片",
        "帮我找{}的图片",
        "{}网站里有什么图",
        "抓取网站{}的图片",
        "获取网页{}的图片",
        "{}页面图片链接",
        "看看{}网站的图",
        "{}有哪些照片",
        "抓取{}的图片",
        "提取网站{}的图片",
        "{}网站的图片有哪些",
        "获取{}的图片列表",
    ]
    
    formats = ["json", "txt"]
    
    for url in urls:
        for _ in range(7):
            template = random.choice(templates)
            fmt = random.choice(formats)
            prompts.append((template.format(url), "get_webpage_images", json.dumps({"url": url, "type": fmt}, ensure_ascii=False)))
    
    return prompts

# ==================== 14. generate_random_ai_image ====================
def generate_random_ai_image():
    """随机生成一张图片"""
    prompts = []
    
    templates = [
        "给我生成一张图片",
        "来张随机图片",
        "生成一张AI图片",
        "随机生成一张图",
        "给我画一张图",
        "AI画一张图",
        "来一张图片",
        "生成一幅图",
        "帮我生成图片",
        "来张有趣的图",
        "随机AI图片",
        "生成一张随机图",
        "给我来张图片",
        "画一张图吧",
        "随机生成一幅图",
        "AI生成图片",
        "生成张图片",
        "来一张AI画",
        "随机图来一张",
        "生成个图片",
    ]
    
    formats = ["txt", "json", "image"]
    
    for template in templates * 5:
        fmt = random.choice(formats)
        args = {"type": fmt} if fmt != "image" else {}
        prompts.append((template, "generate_random_ai_image", json.dumps(args, ensure_ascii=False)))
    
    return prompts

# ==================== 15. check_harassment_phone ====================
def generate_check_harassment_phone():
    """查询是否骚扰电话"""
    prompts = []
    
    numbers = [
        "010-12345678", "021-12345678", "13800138000", "13900139000", "400-123-4567",
        "95588", "95599", "95533", "95555", "10086", "10010", "10000", "12345",
        "18600000000", "15000000000", "13312345678", "18012345678", "17012345678",
        "13912345678", "13812345678", "13612345678", "13512345678", "18812345678",
    ]
    
    templates = [
        "{}是骚扰电话吗",
        "查一下{}是不是骚扰电话",
        "{}是诈骗电话吗",
        "{}这个号码",
        "帮我查{}",
        "{}是不是骗子",
        "{}可信吗",
        "{}靠谱吗",
        "这个号码{}怎么样",
        "查{}是不是骚扰",
        "{}是广告吗",
        "{}有没有问题",
        "看看{}是不是诈骗",
        "{}号码安全吗",
        "{}这个电话能接吗",
        "{}是正规号码吗",
        "查下{}有没有被标记",
        "{}号码怎么样",
        "这个{}是骚扰吗",
        "帮我看看{}",
    ]
    
    for number in numbers:
        for _ in range(4):
            template = random.choice(templates)
            prompts.append((template.format(number), "check_harassment_phone", json.dumps({"tel": number}, ensure_ascii=False)))
    
    return prompts

# ==================== 16. get_random_html_color ====================
def generate_get_random_html_color():
    """随机生成一个颜色"""
    prompts = []
    
    templates = [
        "给我随机生成一个颜色",
        "来一个随机颜色",
        "生成一个颜色",
        "随机颜色",
        "帮我选一个颜色",
        "给我一个颜色",
        "随机生成颜色代码",
        "生成一个HTML颜色",
        "来一个十六进制颜色",
        "随机色",
        "帮我生成颜色",
        "生成随机颜色代码",
        "来一个颜色代码",
        "随机一个颜色",
        "给我个颜色吧",
        "选一个随机颜色",
        "生成个颜色",
        "来随机一个颜色",
        "给我挑一个颜色",
        "随机生成一个色",
    ]
    
    shade_types = ["dark", "light"]
    
    for template in templates * 5:
        shade = random.choice(shade_types)
        prompts.append((template, "get_random_html_color", json.dumps({"type": shade}, ensure_ascii=False)))
    
    return prompts


# 生成所有数据
print("开始生成训练数据...")

training_data.extend(generate_call_phone())
print(f"1. call_phone: {len(generate_call_phone())} 条")

training_data.extend(generate_get_time())
print(f"2. get_time: {len(generate_get_time())} 条")

training_data.extend(generate_get_current_temperature())
print(f"3. get_current_temperature: {len(generate_get_current_temperature())} 条")

training_data.extend(generate_get_lunar_calendar())
print(f"4. get_lunar_calendar: {len(generate_get_lunar_calendar())} 条")

training_data.extend(generate_get_phone_location())
print(f"5. get_phone_location: {len(generate_get_phone_location())} 条")

training_data.extend(generate_get_simp_diary())
print(f"6. get_simp_diary: {len(generate_get_simp_diary())} 条")

training_data.extend(generate_get_gas_price())
print(f"7. get_gas_price: {len(generate_get_gas_price())} 条")

training_data.extend(generate_get_express_info())
print(f"8. get_express_info: {len(generate_get_express_info())} 条")

training_data.extend(generate_get_box_office())
print(f"9. get_box_office: {len(generate_get_box_office())} 条")

training_data.extend(generate_get_license_plate_value())
print(f"10. get_license_plate_value: {len(generate_get_license_plate_value())} 条")

training_data.extend(generate_query_food_calories())
print(f"11. query_food_calories: {len(generate_query_food_calories())} 条")

training_data.extend(generate_get_website_favicon())
print(f"12. get_website_favicon: {len(generate_get_website_favicon())} 条")

training_data.extend(generate_get_webpage_images())
print(f"13. get_webpage_images: {len(generate_get_webpage_images())} 条")

training_data.extend(generate_random_ai_image())
print(f"14. generate_random_ai_image: {len(generate_random_ai_image())} 条")

training_data.extend(generate_check_harassment_phone())
print(f"15. check_harassment_phone: {len(generate_check_harassment_phone())} 条")

training_data.extend(generate_get_random_html_color())
print(f"16. get_random_html_color: {len(generate_get_random_html_color())} 条")

print(f"\n总计: {len(training_data)} 条数据")

# 打乱数据
random.shuffle(training_data)

# 创建DataFrame
df = pd.DataFrame(training_data, columns=["User Prompt", "Tool Name", "Tool Args"])

# 保存为Excel
output_path = "/Users/laurence/working/projects/SiNan/demo/training_demo/generated_training_data.xlsx"
df.to_excel(output_path, index=False)
print(f"\n数据已保存到: {output_path}")

# 显示每个function的统计
print("\n各function数据统计:")
print(df["Tool Name"].value_counts())
