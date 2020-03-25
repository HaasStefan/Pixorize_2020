import numpy as np
from PIL import Image
import cv2
import sys


def output_to_image(net_input, output, name, path):
    l = net_input.cpu().detach().numpy()[0][0]  # l channel
    a = output.cpu().detach().numpy()[0][0]  # a channel
    b = output.cpu().detach().numpy()[0][1]  # b channel

    mapped_l_channel = [[0 for x in range(np.size(l, 1))] for x in range(np.size(l, 0))]
    mapped_a_channel = [[0 for x in range(np.size(a, 1))] for x in range(np.size(a, 0))]
    mapped_b_channel = [[0 for x in range(np.size(b, 1))] for x in range(np.size(b, 0))]

    for j in range(np.size(l, 1)):
        for i in range(np.size(l, 0)):
            mapped_l_channel[i][j] = (((l[i][j] + 1.0) * (255.0)) / (2.0))

    for j in range(np.size(a, 1)):
        for i in range(np.size(a, 0)):
            mapped_a_channel[i][j] = (((a[i][j] + 1.0) * (255.0)) / 2.0)

    for j in range(np.size(b, 1)):
        for i in range(np.size(b, 0)):
            mapped_b_channel[i][j] = (((b[i][j] + 1.0) * (255.0)) / 2.0)

    merged_channels = cv2.merge((np.array(mapped_l_channel, dtype=np.uint8), np.array(mapped_a_channel, dtype=np.uint8),
                                 np.array(mapped_b_channel, dtype=np.uint8)))
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)

    # Image.fromarray(final_image).save(f"../images/output/{now.strftime('%d-%m-%Y %H:%M:%S')}-lab-img{index}.tiff")
    Image.fromarray(final_image).save(f"{path}/{name}.png")



    del l
    del a
    del b
    mapped_l_channel.clear()
    del mapped_l_channel[:]
    mapped_a_channel.clear()
    del mapped_a_channel[:]
    mapped_b_channel.clear()
    del mapped_b_channel[:]
    del merged_channels

