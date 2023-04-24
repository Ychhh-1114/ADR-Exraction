import  pandas as pd


def process_data(file_path):
    data = pd.read_csv(file_path)
    # print(data.columns)
    processed_data = []

    for index, line in data.iterrows():
        dct = {
            "text": [],
            "spo_list": {
                "subject": [],
                "object": [],
                "predicate": [],
            },
        }
        dct["text"] = line["text"]
        dct["spo_list"]["object"] = line["effect"]
        dct["spo_list"]["subject"] = line["drug"]
        dct["spo_list"]["predicate"] = "causes"
        processed_data.append(dct)


    return processed_data








