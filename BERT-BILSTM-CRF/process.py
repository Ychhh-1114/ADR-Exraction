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
def merge_data(data):
    name_list = []
    data_list = []

    for each in data:
        if each["text"] not in name_list:
            name_list.append(each["text"])
            data_list.append(each)
        else:
            index = name_list.index(each["text"])
            data_list[index]["spo_list"].append(each["spo_list"][0])

    return data_list


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
            "subject": "",
            "predicate": "",
            "object": "",
            "subject_type":"drug",
            "object_type":"adverse"

        }

        dct["text"] = line["text"]
        spo["object"] = line["effect"]
        spo["subject"] = line["drug"]
        spo["predicate"] = "causes"
        dct["spo_list"].append(({"subject":spo["subject"],"predicate":spo["predicate"],"object":spo["object"],"subject_type":spo["subject_type"],"object_type":spo["object_type"]}))
        processed_data.append(dct)

    return merge_data(processed_data)




