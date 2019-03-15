import numpy as np
import cv2

import util


class ImageBatchGenerator:
    def __init__(self, train_test_list, frame, spatial, classes, down_sampling_factor, shuffle=True):
        """
        Constructor.
        :param train_test_list: A xxx.txt path which the .txt include all clips for train or test.
        """
        self.train_test_list = train_test_list
        self.pointer = 0
        self.all_clips = []
        self.all_labels = []
        self.frame_num = frame
        self.spatial_size = spatial
        self.all_classes = classes
        self.n_classes = len(classes)
        self.down_sampling_factor = down_sampling_factor
        self.shuffle_state = shuffle

        self.read_train_test_list()
        self.shuffle_data()

    def read_train_test_list(self):
        with open(self.train_test_list) as f:
            clips = f.readlines()
            for clip in clips:
                all_imgs_of_one_clip = self.all_image_of_one_clip(clip.strip())
                self.all_clips.append(all_imgs_of_one_clip)

                label_of_one_clip = self.label2num(clip.split('/')[3])
                self.all_labels.append(label_of_one_clip)

    def all_image_of_one_clip(self, clip_txt):
        all_imgs_of_one_clip = []
        with open(clip_txt) as f:
            imgs = f.readlines()
            for img in imgs:
                all_imgs_of_one_clip.append(img.strip())

        return all_imgs_of_one_clip

    def label2num(self, label_str):
        num = 0
        for one_class in self.all_classes:
            if one_class == label_str:
                return num
            num += 1
        if num == self.n_classes:
            return -1  # error

    def shuffle_data(self):
        """
        Random shuffle the images and labels if shuffle_state=True.
        """
        if self.shuffle_state:
            clips = self.all_clips
            labels = self.all_labels
            self.all_clips = []
            self.all_labels = []

            # create list of permutated index and shuffle data accoding to list
            idx = np.random.permutation(len(labels))
            for ii in idx:
                self.all_clips.append(clips[ii])
                self.all_labels.append(labels[ii])

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into memory.
        :return:
        """
        if self.pointer + batch_size >= self.all_clips.__len__():
            self.pointer = 0
            self.shuffle_data()

        # Get next batch of image (path) and labels
        batch_clips = self.all_clips[self.pointer:self.pointer + batch_size]
        batch_labels = self.all_labels[self.pointer:self.pointer + batch_size]

        # update pointer
        self.pointer += batch_size

        batch_clips_raw = np.ndarray([batch_size, self.frame_num, self.spatial_size[0] // self.down_sampling_factor,
                                      self.spatial_size[1] // self.down_sampling_factor, 3], dtype=float)
        # Expand labels to one hot encoding
        batch_labels_raw = np.zeros((batch_size, self.n_classes), dtype=float)

        # Read batch images.
        for ii in range(batch_size):
            for jj in range(self.frame_num):
                batch_clips_raw[ii][jj] = self.load_image(batch_clips[ii][jj])
            batch_labels_raw[ii][batch_labels[ii]] = 1

        return batch_clips_raw, batch_labels_raw

    def load_image(self, img):
        try:
            img = cv2.imread(img)
        except IOError:
            print("file (%s) not available!!!" % img)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[::self.down_sampling_factor, ::self.down_sampling_factor]  # n times down-sampling
        img = self.per_image_standardization(img)
        return img

    def per_image_standardization(self, img):
        img = img.astype(np.float32)
        # scaling
        img = img - 125.0
        img = img / 255.0
        # normalization
        # img = img - 125.0
        # img = (img - np.mean(img)) / max(np.std(img), 1e-4)

        return img


def main():
    categories = util.CATEGORIES
    frame_num = util.FRAME_NUM
    batch_size = util.BATCH_SIZE

    batch = ImageBatchGenerator('./data/train/train_list.txt', frame_num, [240, 320], categories,
                                down_sampling_factor=8)
    print(batch.all_labels)
    print(len(batch.all_clips))

    batch_clips_raw, batch_labels_raw = batch.next_batch(batch_size)
    print(batch_clips_raw.shape)
    print(batch_labels_raw.shape)

    # for step in range(100):
    #     batch_clip, batch_label = batch.next_batch(batch_size=batch_size)
    #     print(batch_label)
    #     print('=' * 8)


if __name__ == '__main__':
    main()
