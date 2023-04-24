from config import *
from utils import *
from transformers import BertTokenizerFast
from model import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def get_triple_list(sub_head_ids, sub_tail_ids, model, encoded_text, text, mask, offset_mapping):
    id2rel, _ = get_rel()
    triple_list = []
    for sub_head_id in sub_head_ids:
        sub_tail_ids = sub_tail_ids[sub_tail_ids >= sub_head_id]
        if len(sub_tail_ids) == 0:
            continue
        sub_tail_id = sub_tail_ids[0]
        if mask[sub_head_id] == 0 or mask[sub_tail_id] == 0:
            continue
        # 根据位置信息反推出 subject 文本内容
        sub_head_pos_id = offset_mapping[sub_head_id][0]
        sub_tail_pos_id = offset_mapping[sub_tail_id][1]
        subject_text = text[sub_head_pos_id:sub_tail_pos_id]
        # 根据 subject 计算出对应 object 和 relation
        sub_head_seq = torch.tensor(multihot(len(mask), sub_head_id)).to(DEVICE)
        sub_tail_seq = torch.tensor(multihot(len(mask), sub_tail_id)).to(DEVICE)

        pred_obj_head, pred_obj_tail = model.get_objs_for_specific_sub(\
            encoded_text.unsqueeze(0), sub_head_seq.unsqueeze(0), sub_tail_seq.unsqueeze(0))
        # 按分类找对应关系
        pred_obj_head = pred_obj_head[0].T
        pred_obj_tail = pred_obj_tail[0].T
        for j in range(len(pred_obj_head)):
            obj_head_ids = torch.where(pred_obj_head[j] > OBJ_HEAD_BAR)[0]
            obj_tail_ids = torch.where(pred_obj_tail[j] > OBJ_TAIL_BAR)[0]
            for obj_head_id in obj_head_ids:
                obj_tail_ids = obj_tail_ids[obj_tail_ids >= obj_head_id]
                if len(obj_tail_ids) == 0:
                    continue
                obj_tail_id = obj_tail_ids[0]
                if mask[obj_head_id] == 0 or mask[obj_tail_id] == 0:
                    continue
                # 根据位置信息反推出 object 文本内容，mapping中已经有移位，不需要再加1
                obj_head_pos_id = offset_mapping[obj_head_id][0]
                obj_tail_pos_id = offset_mapping[obj_tail_id][1]
                object_text = text[obj_head_pos_id:obj_tail_pos_id]
                triple_list.append((subject_text, id2rel[j], object_text))
    return list(set(triple_list))





if __name__ == '__main__':
    # text = '俞敏洪，出生于1962年9月4日的江苏省江阴市，大学毕业于北京大学西语系。'
    # text = ' We present two children with acute lymphocytic leukemia who developed leukoencephalopathy following administration of a combination of intravenous ara = C and methotrexate during the consolidation phase of chemotherapy.'
    text = "After the chlorambucil was discontinued, the wbc count began to slowly rise and the patient developed clinical AML."
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    tokenized = tokenizer(text, return_offsets_mapping=True)
    info = {}
    info['input_ids'] = tokenized['input_ids']
    info['offset_mapping'] = tokenized['offset_mapping']
    info['mask'] = tokenized['attention_mask']

    input_ids = torch.tensor([info['input_ids']]).to(DEVICE)
    batch_mask = torch.tensor([info['mask']]).to(DEVICE)

    model = torch.load(MODEL_DIR + 'model_48.pth', map_location=DEVICE)

    encoded_text = model.get_encoded_text(input_ids, batch_mask)
    pred_sub_head, pred_sub_tail = model.get_subs(encoded_text)

    sub_head_ids = torch.where(pred_sub_head[0] > SUB_HEAD_BAR)[0]
    sub_tail_ids = torch.where(pred_sub_tail[0] > SUB_TAIL_BAR)[0]
    mask = batch_mask[0]
    encoded_text = encoded_text[0]

    offset_mapping = info['offset_mapping']

    pred_triple_item = get_triple_list(sub_head_ids, sub_tail_ids, model, \
                                       encoded_text, text, mask, offset_mapping)

    print(text)
    print(pred_triple_item)