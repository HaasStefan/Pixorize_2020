import cv2
from torchvision import transforms
from PIL import Image
import random, os
import numpy as np
from skimage.util import random_noise


class DataPreparation:
    # __init__ function for initialisation
    def __init__(self, source_images_dir):
        # Needed directories
        self.source_images_dir = source_images_dir

        # Needed arrays, containing images (color, gray, manipulated)
        self.color_images_array = []
        self.gray_images_array = []
        self.gray_manipulated_images_array = []

        # Create directories
        self.create_directories()

        # Parts
        self.min_part = 0
        self.max_part = 50
        self.range_value = 50

        # Get list of files
        self.list_of_files = self.list_files()

    # Create needed directories
    def create_directories(self):
        # Creates directories, if not exists
        folder = self.source_images_dir
        if not os.path.exists(folder):
            os.mkdir(os.path.join(folder))

    # Get a list with all files - including non image-files - in the directory "source_image_dir"
    def list_files(self):
        list_of_files = list()
        for (dirpath, dirnames, filenames) in os.walk(self.source_images_dir):
            list_of_files += [os.path.join(dirpath, file) for file in filenames]
        random.shuffle(list_of_files)
        return list_of_files

    # Get an array including all color images (RGB)
    def get_color_images(self, list_of_files):
        self.color_images_array.clear()

        index = 0

        for image_file in list_of_files:
            filex_extension = os.path.splitext(image_file)[-1].lower()

            # Testing, if the file ends with ".jpg", ".jpeg", etc.
            if filex_extension == ".jpg" or filex_extension == ".jpeg" or filex_extension == ".png":

                #print("Bei " + image_file + " handelt es sich um eine Bild-Datei!")

                # Loads the image (as Array) and converts from BGR-colorspace to RGB-colorspace
                # (cv2 loads in BGR-colorspace - RGB is needed!)
                np_img = cv2.imread(image_file)
                if np_img is None:
                    break
                color_image = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

                # Test if image is valid and not corrupted (np_img is emtpy)
                if np_img is not None:
                    #print(image_file + " -> korrektes Bild")
                    # Insert image into the "color_images_array"
                    self.color_images_array.append(color_image)

                    # Save image in the directory "color_images_dir" for debug-purposes
                    # Image.fromarray(color_image).save("./images/test/color_images" + "/" + str(index) + ".png")

                #elif np_img is None:
                    #print(image_file + " -> korruptes Bild (fehlerhaft)")

                index += 1

            #else:
                # For debug-purposes
                #print("Bei " + image_file + " handelt es sich um keine Bild-Datei!")

        return self.color_images_array

    # Get an array including all gray images (RGB-colorspace)
    def get_gray_images(self):
        self.gray_images_array.clear()

        index = 0

        for np_img in self.color_images_array:
            # Converting the images into grayscale
            gray_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

            # random gaussian blur and noise
            if random.randrange(2) == 1:
                # Add salt-and-pepper noise to the image.
                noise_img = random_noise(gray_image, mode='s&p', amount=0.3)
                gray_image = np.array(255 * noise_img, dtype='uint8')

            if random.randrange(2) == 1:
                gray_image = cv2.blur(gray_image, (5, 5))

            # Converting the grayscale image into RGB-colorspace and inserting into "gray_images_array"
            gray_image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
            self.gray_images_array.append(gray_image_rgb)

            # Save image in the directory "gray_images_dir" for debug-purposes
            # Image.fromarray(gray_image_rgb).save("./images/gray_images" + "/" + str(index) + ".png")

            index += 1

        return self.gray_images_array

    # Get an array including all manipulated gray images (with RGB-pixels)
    def get_manipulated_gray_images(self):
        self.gray_manipulated_images_array.clear()

        index = 0

        # Inserts random RGB-pixels into the grayscale image
        for manipulated_np_gray_image, np_color_image in zip(self.gray_images_array, self.color_images_array):
            # Get the width and height of the image
            width, height = Image.fromarray(np_color_image).size

            # Based on height and widht, 7 - 10 random RGB-pixels are created
            amount_of_random_pixels = random.randint(10, 20)

            for a in range(0, amount_of_random_pixels):
                random_x_coordinate = random.randint(10, width - 10)
                random_y_coordinate = random.randint(10, height - 10)

                # RGB-values from the original image
                r, g, b = Image.fromarray(np_color_image).getpixel((random_x_coordinate, random_y_coordinate))

                # Now, the RGB-values are read and inserted into the gray-image
                cv2.circle(manipulated_np_gray_image, (random_x_coordinate, random_y_coordinate), 1, (r, g, b), -1)

            # gray images are inserted into the array "gray_manipulated_images_array"
            self.gray_manipulated_images_array.append(manipulated_np_gray_image)

            # Save image in the directory "gray_manipulated_images_array" for debug-purposes
            #Image.fromarray(manipulated_np_gray_image).save("./images/manipulated_images" + "/" + str(index) + ".png")

            index += 1

        return self.gray_manipulated_images_array

    def getAll(self):
        list = []
        # print(numpy.array(self.gray_manipulated_images_array).ndim+1)
        for i in range(0, len(self.gray_manipulated_images_array)):
            # mean std between [-1, 1]
            transform1 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            transform2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
            ])

            # label / Color image
            lab_colour_img = cv2.cvtColor(self.color_images_array[i], cv2.COLOR_RGB2LAB)

            l, a, b = cv2.split(lab_colour_img)

            merged_label = cv2.merge((a, b))
            label = transform2(merged_label).float()

            # Manipualted image
            lab_manipulated_img = cv2.cvtColor(self.gray_manipulated_images_array[i], cv2.COLOR_RGB2LAB)

            l, a, b = cv2.split(lab_manipulated_img)

            merged_manipulated_image = cv2.merge((l, a, b))
            manipulated_image = transform1(merged_manipulated_image).float()

            # tupel #####################################
            tupel = [manipulated_image, label]
            list.append(tupel)

            # save_image(manipulated_image, label, i)

            # print(len(list))

        self.color_images_array.clear()
        self.gray_images_array.clear()
        self.gray_manipulated_images_array.clear()
        self.release(self.color_images_array)
        self.release(self.gray_images_array)
        self.release(self.gray_manipulated_images_array)
        return list

    def load_images_in_parts(self):
        # Only handle 1000 images per function call of "load_images_in_parts"
        ranged_list = self.list_of_files[self.min_part:self.max_part]
        self.get_color_images(ranged_list)
        self.get_gray_images()
        self.get_manipulated_gray_images()

        # Change the parts for the next function call
        # Save old minimum
        old_min_part = self.min_part
        old_max_part = self.max_part

        self.min_part = (old_min_part + self.range_value + 1)
        self.max_part = (old_max_part + self.range_value + 1)

        print("-------------------------------")

        # If the list is empty -> Always return an empty list
        if not (self.color_images_array and self.gray_images_array and self.gray_manipulated_images_array):
            return []
        # If the list is NOT empty -> Return the list of the getAll()-method
        else:
            return self.getAll()

    def release(self, x):
        del x[:]

