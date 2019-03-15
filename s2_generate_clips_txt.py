import os
import glob

import util

train_test_folders = util.TRAIN_TEST_FOLDERS
categories = util.CATEGORIES
frame_num = util.FRAME_NUM


def generate_clips_txt(frame=frame_num):
    for train_test_folder in train_test_folders:
        for category in categories:
            imgs_folders_list = glob.glob(train_test_folder + category + '/*')
            for imgs_folder in imgs_folders_list:
                imgs_folder = imgs_folder.replace('\\', '/')  # Add for Wins
                if imgs_folder.split('.')[-1] != 'avi':  # filter *.avi
                    imgs_list = glob.glob(imgs_folder + '/*.jpg')
                    imgs_list.sort()
                    for clips in range(len(imgs_list) // frame):
                        clip_name = imgs_folder + '_' + str(clips) + '.txt'
                        if os.path.exists(clip_name):
                            os.remove(clip_name)
                        f = open(clip_name, 'w')
                        for ii in range(frame):
                            f.write(imgs_list[clips * frame + ii].replace('\\', '/') + '\n')
                        f.close()


def generate_train_test_txt():
    for train_test_folder in train_test_folders:
        train_test_list = []
        for category in categories:
            txt_list = glob.glob(train_test_folder + category + '/*.txt')
            train_test_list = train_test_list + txt_list

        train_test_list_name = train_test_folder + train_test_folder.split('/')[-2] + '_list.txt'
        if os.path.exists(train_test_list_name):
            os.remove(train_test_list_name)
        f = open(train_test_list_name, 'w')
        for clips in train_test_list:
            f.write(clips.replace('\\', '/') + '\n')
        f.close()


def main():
    generate_clips_txt()
    generate_train_test_txt()


if __name__ == '__main__':
    main()
