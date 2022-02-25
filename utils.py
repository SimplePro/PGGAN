from PIL import Image
from torchvision import transforms
import os
import torch
import numpy as np



class Dataset:

    def __init__(self, directory_list, resolution):
        self.directory_list = directory_list
        self.resolution = resolution


    def image_to_tensor(self, path, res):
        img = Image.open(path).resize(res)

        tensor_img = transforms.ToTensor()(img)
        tensor_img = tensor_img.type(torch.float16)

        return tensor_img


    def dataset_to_tensor(self, directory_path):
        files = os.listdir(directory_path)
        tensor_dataset = torch.zeros((len(files), 3, *self.resolution)).type(torch.float16)

        for i in range(len(files)):
            tensor_dataset[i] = self.image_to_tensor(f"{directory_path}/{files[i]}", self.resolution)
        
        return tensor_dataset


    def extract_dataset(self):
        dataset_pair = []

        for directory_path in self.directory_list:
            dataset_pair.append(self.dataset_to_tensor(directory_path))

        return dataset_pair



def make_gif(paths, save_path, fps=500):
    img, *imgs = [Image.open(path) for path in paths]
    img.save(fp=save_path, format="GIF", append_images=imgs, save_all=True, duration=fps, loop=0)


def merge_test_pred(pred):

    test_size = pred.size(0)
    
    # ex) test_size = 30 -> height = 5, weight = 6
    for i in range(int(np.sqrt(test_size)), test_size+1):
        if test_size % i == 0:
            n_height = max(i, test_size//i)
            n_weight = min(i, test_size//i)
            break
    
    image_size = (
        2048 - (2048 % n_weight),
        2048 - (2048 % n_height)
    )

    one_image_size = (image_size[0] // n_weight, image_size[1] // n_height)

    image = Image.new('RGB', image_size)

    for w in range(n_weight):
        for h in range(n_height):
            img = transforms.ToPILImage()(pred[n_height*w + h])
            img = img.resize(one_image_size)

            image.paste(img, (one_image_size[0] * w, one_image_size[1] * h))
    
    return image



if __name__ == "__main__":
    resolution_list = ["4x4", "8x8", "16x16", "32x32", "64x64", "128x128", "256x256"]
    for resolution in resolution_list:

        # ex) 4x4 -> (4, 4)
        resolution_pair = tuple(map(int, resolution.split("x")))

        dataset = Dataset(
            directory_list=["./dataset/train/cat", "./dataset/train/dog", "./dataset/val/cat", "./dataset/val/dog"],
            resolution=resolution_pair
        )
        
        train_cat, train_dog, valid_cat, valid_dog = dataset.extract_dataset()

        torch.save(train_cat, f"./dataset/{resolution}/train_cat.pt")
        torch.save(train_dog, f"./dataset/{resolution}/train_dog.pt")
        torch.save(valid_cat, f"./dataset/{resolution}/valid_cat.pt")
        torch.save(valid_dog, f"./dataset/{resolution}/valid_dog.pt")

    # no_linear_paths = sorted([int(i.replace("epoch-", "").replace(".jpg", "")) for i in os.listdir("./train_log/no linear layer/")])
    # no_linear_paths = [f"./train_log/no linear layer/epoch-{i}.jpg" for i in no_linear_paths]
    # make_gif(no_linear_paths, "./train_log/no_linear_layer.gif", 5)
