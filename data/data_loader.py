from os.path import join

from dataset_medical_resize2020 import DatasetFromFolder


def get_training_set(root_dir, datast):
    train_dir =root_dir+datast

    return DatasetFromFolder(train_dir)


def get_test_set(root_dir, datast):
    #test_dir = join(root_dir, "test")
    test_dir=root_dir+datast

    return DatasetFromFolder(test_dir)
