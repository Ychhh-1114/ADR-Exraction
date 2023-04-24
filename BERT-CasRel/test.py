from utils import *
from model import *
from torch.utils import data
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
    model = torch.load(MODEL_DIR + f'model_48.pth', map_location=DEVICE)

    dataset = Dataset('test')

    with torch.no_grad():

        loader = data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)

        correct_num, predict_num, gold_num = 0, 0, 0
        pred_triple_list = []
        true_triple_list = []

        for b, (batch_mask, batch_x, batch_y) in enumerate(loader):
            batch_text, batch_sub_rnd = batch_x
            batch_sub, batch_obj_rel = batch_y

            # 整理input数据并预测
            input_mask = torch.tensor(batch_mask).to(DEVICE)
            input = (
                torch.tensor(batch_text['input_ids']).to(DEVICE),
                torch.tensor(batch_sub_rnd['head_seq']).to(DEVICE),
                torch.tensor(batch_sub_rnd['tail_seq']).to(DEVICE),
            )
            encoded_text, pred_y = model(input, input_mask)

            # 整理target数据并计算损失
            true_y = (
                torch.tensor(batch_sub['heads_seq']).to(DEVICE),
                torch.tensor(batch_sub['tails_seq']).to(DEVICE),
                torch.tensor(batch_obj_rel['heads_mx']).to(DEVICE),
                torch.tensor(batch_obj_rel['tails_mx']).to(DEVICE),
            )
            loss = model.loss_fn(true_y, pred_y, input_mask)

            print('>> batch:', b, 'loss:', loss.item())

            # 计算关系三元组，和统计指标
            pred_sub_head, pred_sub_tail, _, _ = pred_y
            true_triple_list += batch_text['triple_list']

            # 遍历batch
            for i in range(len(pred_sub_head)):
                text = batch_text['text'][i]
                true_triple_item = true_triple_list[i]
                mask = batch_mask[i]
                offset_mapping = batch_text['offset_mapping'][i]

                sub_head_ids = torch.where(pred_sub_head[i] > SUB_HEAD_BAR)[0]
                sub_tail_ids = torch.where(pred_sub_tail[i] > SUB_TAIL_BAR)[0]

                pred_triple_item = get_triple_list(sub_head_ids, sub_tail_ids, model, \
                                                   encoded_text[i], text, mask, offset_mapping)

                # 统计个数
                correct_num += len(set(true_triple_item) & set(pred_triple_item))
                predict_num += len(set(pred_triple_item))
                gold_num += len(set(true_triple_item))

                pred_triple_list.append(pred_triple_item)

        precision = correct_num / (predict_num + EPS)
        recall = correct_num / (gold_num + EPS)
        f1_score = 2 * precision * recall / (precision + recall + EPS)
        print('\tcorrect_num:', correct_num, 'predict_num:', predict_num, 'gold_num:', gold_num)
        print('\tprecision:%.3f' % precision, 'recall:%.3f' % recall, 'f1_score:%.3f' % f1_score)