import os
from tqdm import tqdm
root_path = '/data/disk_a/xujl/Project/FSDE/tools/LoFTR/save_matching_points/nuScenes/6Camera_7Frame/USAC_ACCURATE/'
for num in tqdm(range(5,30)):
    num = '%06d'%num
    if not os.path.isfile(root_path + num + '/num.txt'):
        continue

    cmd = 'cd ../../tools/ceres/build && ./ceres_example %s'%num

    a = os.system(cmd)

