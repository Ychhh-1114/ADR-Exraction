from flask import Blueprint,request,jsonify,render_template
import json
from biobert_casrel_model.model import CasRel
from biobert_casrel_model.predict import predict

from models import ADR,ReADR,Trans


bp = Blueprint("model",__name__,url_prefix="/model")



def res2ADR(res,flag):
    adr_list = []
    adr_name_list = []
    for each in res:
        if flag == 1:
            drug = trans_en2ch(each[0])
            adverse = trans_en2ch(each[2])
        else:
            drug = each[0]
            adverse = each[2]

        if drug not in adr_name_list:
            adr = ADR(drug, adverse)
            adr_list.append(adr)
            adr_name_list.append(drug)
        else:
            index = adr_name_list.index(drug)
            adr = adr_list[index]
            if adverse not in adr.advers:
                adr.add_adverse(adverse)

    return objs2json(adr_list)

def objs2json(objs):
    obj_list = []
    for obj in objs:
        obj_list.append(obj.__dict__)

    return obj_list



def trans_ch2en(text):
    trans = Trans(text,"zh")
    return trans.trans()

def trans_en2ch(text):
    trans = Trans(text,"en")
    return trans.trans()

# @bp.route("/extract")
# def demo():
#     return render_template("/about.html")

@bp.route("/extract",methods=["POST"])
def process_data():
    data = request.get_data()
    jsonStr = json.loads(data.decode("utf-8"))
    data = jsonStr["text"]
    lang = jsonStr["lang"]

    flag = 0

    if lang == "中文":
        data = trans_ch2en(data)
        flag = 1
    # print(data)
    pred_triple_item = predict(data)
    # print(lang)
    adr_list = res2ADR(pred_triple_item,flag)

    if pred_triple_item == []:
        readr = ReADR(adr_list,0)
    else:
        readr = ReADR(adr_list,1)
    # print(readr.adr_list)
    return jsonify(readr.__dict__)




