import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import ImageTk

from keras.preprocessing.image import image
from keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import filedialog

def train_model():
        #Training Image Processing
        train_data = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        training_set = train_data.flow_from_directory('train',target_size=(64,64),batch_size=32,class_mode='categorical')

        test_data = ImageDataGenerator(rescale=1./255)
        test_set = test_data.flow_from_directory('test',target_size=(64,64),batch_size=32,class_mode='categorical')

        #Building Model
        cnn = tf.keras.models.Sequential()

        cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[64,64,3]))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

        cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

        cnn.add(tf.keras.layers.Dropout(0.5))
        cnn.add(tf.keras.layers.Flatten())

        cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
        cnn.add(tf.keras.layers.Dense(units=6, activation='softmax'))

        cnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

        cnn.fit(x=training_set,validation_data=test_set,epochs=80)

        return cnn

def get_file(cnn):

    file = filedialog.askopenfile(mode='rb', filetypes=[('image Files', '*.')])

    if file:

        filePath = os.path.abspath(file.name)
        test_image = image.load_img(str(filePath), target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)

        global label
        global info1
        global info2
        global my_img_label

        if result[0][0] == 1:

            my_img = ImageTk.PhotoImage(Image.open("flowerImages/cirsuimvulgare.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()

            label = tk.Label(root, text="Cirsuim Vulgare", font=("Arial", 25))
            label.pack()

            info1 = tk.Label(root, text="Bull thistle is a moderately large wildflower measuring approximately 60-180 cm. in height.")
            info1.pack()
            info2 = tk.Label(root, text="It is most easily recognized by its spine-covered fruit and bright pink or white flower cluster positioned atop the terminal shoot")
            info2.pack()

        elif result[0][1] == 1:

            my_img = ImageTk.PhotoImage(Image.open("flowerImages/daisy.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()


            label = tk.Label(root, text="Daisy", font=("Arial", 25))
            label.pack()

            info1 = tk.Label(root, text="Though they originated in Europe and temperate regions of Asia, daisies eventually were introduced to Australia and North America.")
            info1.pack()
            info2 = tk.Label(root, text="They're now found on every continent except for Antarctica")
            info2.pack()


        elif result[0][2] == 1:

            my_img = ImageTk.PhotoImage(Image.open("flowerImages/dandelion.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()

            label = tk.Label(root, text="Dandelion", font=("Arial", 25))
            label.pack()



            info1 = tk.Label(root,text="Dandelion – the ubiquitous weed – is a flowering plant belonging to the Asteraceae family")
            info1.pack()
            info2 = tk.Label(root, text=" Considered as a native to Mediterranean, dandelion plants were known quite well by ancient Egyptians, Romans, and Greeks")
            info2.pack()

        elif result[0][3] == 1:

            my_img = ImageTk.PhotoImage(Image.open("flowerImages/rose.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()

            label = tk.Label(root, text="Rose", font=("Arial", 25))
            label.pack()


            info1 = tk.Label(root, text="Archaeologists have discovered rose fossils that date back 35 million years")
            info1.pack()
            info2 = tk.Label(root, text="Originated in Central Asia but spread and grew wild over nearly the entire northern hemisphere")
            info2.pack()

        elif result[0][4] == 1:

            my_img = ImageTk.PhotoImage(Image.open("flowerImages/sunflower.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()


            label = tk.Label(root, text="Sunflower", font=("Arial", 25))
            label.pack()


            info1 = tk.Label(root, text="Helianthus is a genus comprising about 70 species of annual and perennial flowering plants in the daisy family Asteraceae")
            info1.pack()
            info2 = tk.Label(root, text="A universal fact most people know is sunflowers are yellow. Sunflowers can even be red and purple!")
            info2.pack()


        elif result[0][5] == 1:

            my_img = ImageTk.PhotoImage(Image.open("flowerImages/tulip.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()

            label = tk.Label(root, text="Tulip", font=("Arial", 25))
            label.pack()

            info1 = tk.Label(root, text="Tulips are a genus of spring-blooming perennial herbaceous bulbiferous geophytes")
            info1.pack()
            info2 = tk.Label(root, text="Originally growing wild in the valleys of the Tian Shan Mountains, tulips were cultivated in Constantinople as early as 1055")
            info2.pack()
        else:
            label = tk.Label(root, text="NOT RECOGNIZED.TRY ANOTHER IMAGE!!!!", font=("Arial", 25))
            label.pack()

def clear_fun():
    label.pack_forget()
    info1.pack_forget()
    info2.pack_forget()
    my_img_label.pack_forget()


cnn = train_model()

root = tk.Tk()
root.title("ODDISH Flower Recognition")
root.geometry('1300x700')

img = Image.open("logo.ico")
resize_image = img.resize((200, 200))
logo_img = ImageTk.PhotoImage(resize_image)
logo_img_label = tk.Label(root, image=logo_img)
logo_img_label.pack()
logo_img_label.image = logo_img
logo_img_label.pack()

intro_label = tk.Label(root,
                       text = "Welome to ODDISH. Please upload your flower image and we will tell you the name and some cool facts!",
                       font=("Arial", 25))
intro_label.pack()

tk.Button(root, text="upload", command= lambda: get_file(cnn)).pack()
tk.Button(root, text="clear", command = clear_fun).pack()

root.mainloop()

























