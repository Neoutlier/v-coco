import __init__
import vsrl_utils as vu
import numpy as np
import copy
import os
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt

print 1
vcoco_all = vu.load_vcoco('vcoco_train')
classes = [x['action_name'] for x in vcoco_all]
action_dict = {}
role_dict = {}
for i, x in enumerate(vcoco_all):
    # print '{:>20s}'.format(x['action_name']), x['role_name']
    action_dict[i+1] = x['action_name']
    role_dict[i+1] = x['role_name']
new_dict = {27:7, 28:8, 29:15}
print action_dict
print role_dict

sqrt = math.sqrt
def abboxdist(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    xx1, yy1, xx2, yy2 = bbox2
    dist = sqrt((x1-xx1) ** 2 + (y1-yy1) ** 2) + sqrt((x2-xx2) ** 2 + (y2-yy2) ** 2)
    return dist

result_file = 'results.txt.all'
assert(os.path.exists(result_file))
prediction_list = []
pred_dict = {}
with open(result_file, 'r') as f:
    for line in f:
        line2 = line[:-1]
        elements = line2.split(' ')
        if len(elements) == 12:
            imgid = elements[0].rsplit('/')[-1]
            imgid = int(imgid.split('.')[0])
            person_box = [float(elements[1]), float(elements[2]), float(elements[3]), float(elements[4])]
            score = float(elements[-1])
            if score < 0.05:
                continue
            found = False
            for agent_ins in prediction_list:
                if imgid == agent_ins['image_id'] and abboxdist(agent_ins['person_box'], person_box) < 1:
                    new_pred_dict = agent_ins
                    found = True
                    continue
            if not found:
                new_pred_dict = copy.deepcopy(pred_dict)
                for aid in range(1,27):
                    an = action_dict[aid]
                    for role in role_dict[aid]:
                        k = an + '_' + role
                        if k not in new_pred_dict:
                            if role == 'agent':
                                new_pred_dict[k] = 0
                            else:
                                new_pred_dict[k] = [0, 0, 0, 0, 0]
            obj_box = [float(elements[6]), float(elements[7]), float(elements[8]), float(elements[9]), score]
            action_kind = int(elements[-2])
            rn = 1
            if action_kind > 26:
                rn =  2
                action_kind = new_dict[action_kind]
            keyone = action_dict[action_kind] + '_agent'
            if action_kind > 26 or len(role_dict[action_kind]) > 1:
                keytwo = action_dict[action_kind] + '_' + role_dict[action_kind][rn]
                if new_pred_dict[keytwo][4] > score:
                    #print new_pred_dict[keytwo], obj_box
                    continue
                new_pred_dict[keytwo] = obj_box
            elif score > 0.1:
                print action_dict[action_kind], score
            new_pred_dict['person_box'] = person_box
            new_pred_dict[keyone] = score
            new_pred_dict['image_id'] = imgid
            if not found:
                prediction_list.append(new_pred_dict)


pickle_out = open("detections.pkl","wb")
pickle.dump(prediction_list, pickle_out)
pickle_out.close()

from vsrl_eval import VCOCOeval

vsrl_annot_file = 'data/vcoco/vcoco_val.json'
coco_file = 'data/instances_vcoco_all_2017.json'
split_file = 'data/splits/vcoco_val.ids'
det_file = 'detections.pkl'
vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)

fp, fn =vcocoeval._do_eval(det_file, ovr_thresh=0.5)

pickle_out = open("fp_fn_samples.pkl","wb")
pickle.dump([fp, fn], pickle_out)
pickle_out.close()
