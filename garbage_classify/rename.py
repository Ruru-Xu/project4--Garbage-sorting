import os
img_dir = 'additional_train_data'
	for root, dirs, files in os.walk(img_dir):
        dir_list = os.listdir(root)
        for dir in dir_list:
            img_list = os.listdir(os.path.join(root_path, dir))
            print(dir, len(img_list))

oot_path = '../../garbage_classify/additional_train_data'
    dir_list = os.listdir(root_path)
    print(dir_list)
    for dir in dir_list:
        img_list = os.listdir(os.path.join(root_path, dir))
        print(dir, len(img_list))
        for idx, img_name in enumerate(img_list):
            old_path = os.path.join(root_path, dir, img_name)
            new_path = os.path.join(root_path, dir+'_' +dir, str(idx) + '.jpg')
            os.renames(old_path, new_path)