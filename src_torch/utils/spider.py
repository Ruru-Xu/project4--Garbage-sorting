# -*- coding:utf-8 -*-
import os
import re
import urllib
import json
import socket
import urllib.request
import urllib.parse
import urllib.error
# 设置超时
import time

timeout = 5
socket.setdefaulttimeout(timeout)


class Crawler:
    # 睡眠时长
    __time_sleep = 0.1
    __amount = 0
    __start_amount = 0
    __counter = 0

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    # 获取图片url内容等
    # t 下载图片时间间隔
    def __init__(self, t=0.1, key='-1'):
        self.time_sleep = t
        self.key = key
        self.save_dir = "../../garbage_classify/spider_train_data/"
    # 获取后缀名
    def get_suffix(self, name):
        m = re.search(r'\.[^\.]*$', name)
        if m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.jpeg'

    # 获取referrer，用于生成referrer
    def get_referrer(self, url):
        par = urllib.parse.urlparse(url)
        if par.scheme:
            return par.scheme + '://' + par.netloc
        else:
            return par.netloc

        # 保存图片
    def save_image(self, rsp_data, word):
        word = self.key+'_'+word
        if not os.path.exists(self.save_dir + word):
            os.makedirs(self.save_dir + word)
        # 判断名字是否重复，获取图片长度
        self.__counter = len(os.listdir(self.save_dir + word)) + 1
        for image_info in rsp_data['imgs']:

            try:
                time.sleep(self.time_sleep)
                suffix = self.get_suffix(image_info['objURL'])
                # 指定UA和referrer，减少403
                refer = self.get_referrer(image_info['objURL'])
                opener = urllib.request.build_opener()
                opener.addheaders = [
                    ('User-agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0'),
                    ('Referer', refer)
                ]
                urllib.request.install_opener(opener)
                # 保存图片
                urllib.request.urlretrieve(image_info['objURL'], self.save_dir + word + '/' + str(self.__counter) + str(suffix))
            except urllib.error.HTTPError as urllib_err:
                print(urllib_err)
                continue
            except Exception as err:
                time.sleep(1)
                print(err)
                print("产生未知错误，放弃保存")
                continue
            else:
                print("+1,已有" + str(self.__counter) + "张图")
                self.__counter += 1
        return

    # 开始获取
    def get_images(self, word):
        search = urllib.parse.quote(word)
        # pn int 图片数
        pn = self.__start_amount
        while pn < self.__amount:

            url = 'http://image.baidu.com/search/avatarjson?tn=resultjsonavatarnew&ie=utf-8&word=' + search + '&cg=girl&pn=' + str(
                pn) + '&rn=60&itg=0&z=0&fr=&width=&height=&lm=-1&ic=0&s=0&st=-1&gsm=1e0000001e'
            # 设置header防ban
            try:
                time.sleep(self.time_sleep)
                req = urllib.request.Request(url=url, headers=self.headers)
                page = urllib.request.urlopen(req)
                rsp = page.read().decode('unicode_escape')
            except UnicodeDecodeError as e:
                print(e)
                print('-----UnicodeDecodeErrorurl:', url)
            except urllib.error.URLError as e:
                print(e)
                print("-----urlErrorurl:", url)
            except socket.timeout as e:
                print(e)
                print("-----socket timout:", url)
            else:
                # 解析json
                rsp_data = json.loads(rsp)
                self.save_image(rsp_data, word)
                # 读取下一页
                print("下载下一页")
                pn += 60
            finally:
                page.close()
        print("下载任务结束")
        return

    def start(self, word, spider_page_num=1, start_page=1):
        """
        爬虫入口
        :param word: 抓取的关键词
        :param spider_page_num: 需要抓取数据页数 总抓取图片数量为 页数x60
        :param start_page:起始页数
        :return:
        """
        self.__start_amount = (start_page - 1) * 60
        self.__amount = spider_page_num * 60 + self.__start_amount
        self.get_images(word)


id_name = {
    # "0": "其他垃圾/打包盒",
    # "1": "其他垃圾/大号塑料袋",
    # "2": "其他垃圾/烟蒂照片",
    # "3": "其他垃圾/牙签",
    # "4": "其他垃圾/碎花盆",
    # "5": "其他垃圾/一双竹筷",
    # "6": "厨余垃圾/剩饭剩菜",
    # "7": "厨余垃圾/大棒骨头",
    # "8": "厨余垃圾/果皮", #水果皮",
    # "9": "厨余垃圾/烂水果",
    # "10": "厨余垃圾/茶叶渣",
    # "11": "厨余垃圾/一捆蔬菜", # 烂菜叶菜根",
    # "12": "厨余垃圾/鸡蛋壳",
    # "13": "厨余垃圾/鱼骨头",
    # "14": "可回收物/充电宝",
    # "15": "可回收物/背包",
    # "16": "可回收物/化妆品瓶",
    # "17": "可回收物/塑料玩具",
    # "18": "可回收物/水盆",
    # "19": "可回收物/塑料衣架",
    # "20": "可回收物/快递包裹纸袋",
    # "21": "可回收物/插头电线",
    # "22": "可回收物/一件旧衣服",
    # "23": "可回收物/易拉罐",
    # "24": "可回收物/枕头",
    # "25": "可回收物/毛绒玩具",
    # "26": "可回收物/一瓶沐浴露",
    # "27": "可回收物/瓷杯",
    # "28": "可回收物/皮鞋",
    # "29": "可回收物/实木菜板", # 发霉的菜板",#旧菜板",
    # "30": "可回收物/纸箱",
    # "31": "可回收物/调料瓶",
    "32": "可回收物/青岛玻璃酒瓶", #二锅头",
    # "33": "可回收物/铁罐铁瓶",
    # "34": "可回收物/空铁锅电饭锅",
    # "35": "可回收物/透明空油桶",
    # "36": "可回收物/饮料瓶",
    # "37": "有害垃圾/干电池",
    # "38": "有害垃圾/软膏",
    # "39": "有害垃圾/药盒包装", # 纸药盒"
}

if __name__ == '__main__':

    for key in id_name.keys():
        crawler = Crawler(0.05, key)  # 抓取延迟为 0.05
        crawler.start(id_name[key].split('/')[1], 4, 1)

    # 抓取关键词为 “二次元 美女”，总数为 10 页（即总共 10*60=600 张），起始抓取的页码为 1
    # crawler.start('帅哥', 5)  # 抓取关键词为 “帅哥”，总数为 5 页（即总共 5*60=300 张）
