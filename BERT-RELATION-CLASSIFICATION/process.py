import  pandas as pd


# def process_data(file_path):
#     data = pd.read_csv(file_path)
#     # print(data.columns)
#     processed_data = []
#
#     for index, line in data.iterrows():
#         dct = {
#             "text": [],
#             "spo_list": {
#                 "subject": [],
#                 "object": [],
#                 "predicate": [],
#             },
#         }
#         dct["text"] = line["text"]
#         dct["spo_list"]["object"] = line["effect"]
#         dct["spo_list"]["subject"] = line["drug"]
#         dct["spo_list"]["predicate"] = "causes"
#         processed_data.append(dct)
#
#
#     return processed_data
#
#
#



def process_data(file_path):
    data = pd.read_csv(file_path)
    # print(data.columns)
    processed_data = []

    for index, line in data.iterrows():
        dct = {
            "text": [],
            "spo_list": [],
        }
        spo = {
            "text":"",
            "ent1": "",
            "ent2": "",
            "rel": ""
        }

        spo["text"] = line["text"]
        spo["ent2"] = line["effect"]
        spo["ent1"] = line["drug"]
        spo["rel"] = "causes"
        processed_data.append(spo)

    return processed_data




