from model import AE
import torch
from output_to_image import output_to_image
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader

model_path = "./saves/2019-12-31 14_21_53-model-68000.pth"
img = "xxxx"    # Es wird zwar ein rgb Bild erwartet, jedoch muss dieses zuerst einmal in Grayscale umgewandelt worden sein und anschließend wieder in ein rgb Bild
output_folder = "xxxx"
name = "test_image"


def main():
    model = AE()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.eval()

    np_img = cv2.imread(img)
    image = cv2.cvtColor(np_img, cv2.COLOR_BGR2LAB)

    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input = transform1(image).float()
    tuple = [input, 0]
    list = []
    list.append(tuple)

    dataloader = DataLoader(list, batch_size=1, shuffle=False)
    for batch in dataloader:
        in_img, _ = batch
        output = model(in_img)
        output_to_image(in_img, output, name, output_folder)


if __name__ == "__main__":
    main()

