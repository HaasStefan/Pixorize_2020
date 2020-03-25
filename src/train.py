import data_preparation as prep
from model import AE
from output_to_image import output_to_image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import os
from colorama import Fore


input_folder = "./images/aigner"
output_folder = "./images/output"
test_folder = "./images/ball"
lr = 0.0001


def weights_init(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def test(model, time):
    path = os.path.join(output_folder, f"{time}")
    os.mkdir(path)

    index = 0
    datapreparer = prep.DataPreparation(test_folder)
    while True:
        data = datapreparer.load_images_in_parts()
        print("inside test")

        if len(data) == 0:
            break

        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for batch in dataloader:
            image, label = batch

            output = model(image)
            output_to_image(image, label, index, path)

            print(index)
            index += 1
        data.clear()
        del data[:]
        del dataloader

    print("outside test")


def main():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    path = os.path.join(output_folder, f"{now}")
    os.mkdir(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AE().to(device=device)
    model.apply(weights_init)

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_loss = []
    local_loss = []
    index = 0

    datapreparer = prep.DataPreparation(input_folder)

    while True:
        data = datapreparer.load_images_in_parts()

        if len(data) == 0:
            print("break train")
            break

        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for batch in dataloader:
            try:
                image, label = batch
                image = image.to(device=device)
                label = label.to(device=device)

                output = model(image).to(device=device)
                output_to_image(image, label, index, path)

                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()  # calculate gradients
                optimizer.step()  # update weights

                local_loss.append(loss.item())

                if index % 500 == 0:
                    global_loss.append(sum(local_loss) / len(local_loss))
                    local_loss.clear()

                print(Fore.WHITE + f"{index}\tloss: {loss.item()}")

            except:
                print(Fore.RED + f"{index}\t An Error occured!")
            index += 1

    print("outside train")
    # plot all train_losses
    plt.plot(global_loss)
    plt.title('AE')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.show()

    # save plot
    plt.savefig(f"./saves/{now} plot.pdf")

    global_loss.clear()
    del global_loss[:]

    # save model
    torch.save(model.state_dict(), f"./saves/{now} model.pth")

    # test
    test(model=model, time=now)

    print(Fore.BLUE + f"{now}")


if __name__ == "__main__":
    main()
