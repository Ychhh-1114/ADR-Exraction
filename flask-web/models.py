import hashlib
import http.client
import json
import random
import urllib


class ADR():
    def __init__(self,drug,adverse):
        self.drug = drug
        self.advers = []
        self.add_adverse(adverse)

    def add_adverse(self,adverse):
        self.advers.append(adverse)



class ReADR():
    def __init__(self,adr_list,key):
        self.nullkey = key
        self.adr_list = adr_list


class Trans():
    def __init__(self,text,lang):
        self.text = text
        if lang == "en":
            self.fromlang = lang
            self.tolang = "zh"
        else:
            self.fromlang = lang
            self.tolang = "en"

    def trans(self):
        def baiduTranslate(translate_text):
            '''
            :param translate_text: 待翻译的句子，len(q)<2000
            :param flag: 1:原句子翻译成英文；0:原句子翻译成中文
            :return: 返回翻译结果。
            For example:
            q=我今天好开心啊！
            result = {'from': 'zh', 'to': 'en', 'trans_result': [{'src': '我今天好开心啊！', 'dst': "I'm so happy today!"}]}
            '''

            appid = '20230209001556045'  # 填写你的appid
            secretKey = 'J1b4B24XzdV4VNhL3JAI'  # 填写你的密钥
            httpClient = None
            myurl = '/api/trans/vip/translate'  # 通用翻译API HTTP地址
            fromLang = self.fromlang  # 原文语种
            toLang = self.tolang
            # if flag:
            #     toLang = 'en'  # 译文语种
            # else:
            #     toLang = 'zh'  # 译文语种

            salt = random.randint(3276, 65536)

            sign = appid + translate_text + str(salt) + secretKey
            sign = hashlib.md5(sign.encode()).hexdigest()
            myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(translate_text) + '&from=' + fromLang + \
                    '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

            # 建立会话，返回结果
            try:
                httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
                httpClient.request('GET', myurl)
                # response是HTTPResponse对象
                response = httpClient.getresponse()
                result_all = response.read().decode("utf-8")
                result = json.loads(result_all)
                print(result)
                # return result
                return result['trans_result'][0]['dst']

            except Exception as e:
                print(e)
            finally:
                if httpClient:
                    httpClient.close()



        return baiduTranslate(self.text)




