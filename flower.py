import tensorflow as tf
import numpy as np
import tkinter as tk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import ImageTk
from keras.preprocessing.image import image
from keras.models import load_model
from tkinter import filedialog



#Function performs data processing on test and train images and train CNN models using keras activation functions 
def train_model():
     
        #Perform Imagse Data Processing and change configuration of train inputs. (rescale) 
        train_data = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        #Apply new train_data configurations to Train images. Set target size, bath size, and classify images based on the label on the directory the images live
        training_set = train_data.flow_from_directory('train',target_size=(64,64),batch_size=32,class_mode='categorical')

        #Perform Imagse Data Processing and change configuration of test inputs. (rescale) 
        test_data = ImageDataGenerator(rescale=1./255)
        
        #Apply new test_data configurations to test images. Set target size, bath size, and classify images based on the label on the directory the images live
        test_set = test_data.flow_from_directory('test',target_size=(64,64),batch_size=32,class_mode='categorical')

        #Building Model
        cnn = tf.keras.models.Sequential()

        #add a convolution neuaral Network layer Params: filters, kernal_size, relu function, image shape. 
        #relu activation functions: https://keras.io/api/layers/activations/
        cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[64,64,3]))
        
        #Using the max pooling function
        #add a pooling layer Prams: set pool size and stride moves on image pixel
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
        
        #Image is is narrowed by strong features 
        #Add another convolution layer with relu activation function
        cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
        
        #Add another pooling layer using the max pooling function 
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

        
        #Add a dropout layer to preven overfitting
        #https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
        cnn.add(tf.keras.layers.Dropout(0.5))
        
        #Convert layer into a 1-D Array
        cnn.add(tf.keras.layers.Flatten())

        # fully connected layer: https://keras.io/api/layers/core_layers/dense/
        #set to 128 nuerons and relu activation
        cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
        
        #add layer and perform probability distribution to the images classes (6 flower classes) 
        cnn.add(tf.keras.layers.Dense(units=6, activation='softmax'))

        
        #Call the compile function to compiled the CNN model. 
        #optimize model using 'rmsprop' algorthm : The centered version additionally maintains a moving average of the gradients, 
        #and uses that average to estimate the variance.
        #https://keras.io/api/optimizers/rmsprop/
        #use "categorical_crossentropy to computes the crossentropy loss between the labels and predictions
        #https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class
        cnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        
        
        #train the model for a fixed number of iterations (epochs)
        #set the training and validations set to newly rescaled test and train sets
        cnn.fit(x=training_set,validation_data=test_set,epochs=80)

        #return the model (CNN object)
        return cnn

#Backend code for prediction input
def get_file(cnn):

    #File upload, accept/open all typed of files using 'rb" mode
    file = filedialog.askopenfile(mode='rb', filetypes=[('image Files', '*.')])

    if file:
        
        #get filed path of uploaded images 
        filePath = os.path.abspath(file.name)
        
        #cast str to file path
        #set user upload image target size
        test_image = image.load_img(str(filePath), target_size=(64, 64))
        
        #convert image to numpy array using the Pillow image library 
        test_image = image.img_to_array(test_image)
        
        #expand array 
        test_image = np.expand_dims(test_image, axis=0)
        
        #Get prediction on CNN model using test_image
        result = cnn.predict(test_image)

        #Set Tkinter labels global variables 
        global label
        global info1
        global info2
        global my_img_label

        #if prediciton is Bull thistle 
        if result[0][0] == 1:

           #display image on tkinter window 
            my_img = ImageTk.PhotoImage(Image.open("flowerImages/cirsuimvulgare.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()

            #diplay flower facts and name on tkinter window 
            label = tk.Label(root, text="Cirsuim Vulgare", font=("Arial", 25))
            label.pack()

            info1 = tk.Label(root, text="Bull thistle is a moderately large wildflower measuring approximately 60-180 cm. in height.")
            info1.pack()
            info2 = tk.Label(root, text="It is most easily recognized by its spine-covered fruit and bright pink or white flower cluster positioned atop the terminal shoot")
            info2.pack()

        #if prediciton is daisy
        elif result[0][1] == 1:

          #display image on tkinter window 
            my_img = ImageTk.PhotoImage(Image.open("flowerImages/daisy.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()


             #diplay flower facts and name on tkinter window 
            label = tk.Label(root, text="Daisy", font=("Arial", 25))
            label.pack()

            info1 = tk.Label(root, text="Though they originated in Europe and temperate regions of Asia, daisies eventually were introduced to Australia and North America.")
            info1.pack()
            info2 = tk.Label(root, text="They're now found on every continent except for Antarctica")
            info2.pack()

        #if prediciton is dandelion
        elif result[0][2] == 1:

             #display image on tkinter window 
            my_img = ImageTk.PhotoImage(Image.open("flowerImages/dandelion.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()
              
            #diplay flower facts and name on tkinter window 
            label = tk.Label(root, text="Dandelion", font=("Arial", 25))
            label.pack()

            info1 = tk.Label(root,text="Dandelion – the ubiquitous weed – is a flowering plant belonging to the Asteraceae family")
            info1.pack()
            info2 = tk.Label(root, text=" Considered as a native to Mediterranean, dandelion plants were known quite well by ancient Egyptians, Romans, and Greeks")
            info2.pack()

        #if prediciton is rose
        elif result[0][3] == 1:

            #display image on tkinter window 
            my_img = ImageTk.PhotoImage(Image.open("flowerImages/rose.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()

            #diplay flower facts and name on tkinter window 
            label = tk.Label(root, text="Rose", font=("Arial", 25))
            label.pack()

            info1 = tk.Label(root, text="Archaeologists have discovered rose fossils that date back 35 million years")
            info1.pack()
            info2 = tk.Label(root, text="Originated in Central Asia but spread and grew wild over nearly the entire northern hemisphere")
            info2.pack()

        #if prediciton is sunflower
        elif result[0][4] == 1:

            #display image on tkinter window 
            my_img = ImageTk.PhotoImage(Image.open("flowerImages/sunflower.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()

             #diplay flower facts and name on tkinter window 
            label = tk.Label(root, text="Sunflower", font=("Arial", 25))
            label.pack()

            info1 = tk.Label(root, text="Helianthus is a genus comprising about 70 species of annual and perennial flowering plants in the daisy family Asteraceae")
            info1.pack()
            info2 = tk.Label(root, text="A universal fact most people know is sunflowers are yellow. Sunflowers can even be red and purple!")
            info2.pack()

        #if prediciton is tulip
        elif result[0][5] == 1:

            #display image on tkinter window 
            my_img = ImageTk.PhotoImage(Image.open("flowerImages/tulip.png"))
            my_img_label = tk.Label(root, image=my_img)
            my_img_label.pack()
            my_img_label.image = my_img
            my_img_label.pack()

            #diplay flower facts and name on tkinter window 
            label = tk.Label(root, text="Tulip", font=("Arial", 25))
            label.pack()

            info1 = tk.Label(root, text="Tulips are a genus of spring-blooming perennial herbaceous bulbiferous geophytes")
            info1.pack()
            info2 = tk.Label(root, text="Originally growing wild in the valleys of the Tian Shan Mountains, tulips were cultivated in Constantinople as early as 1055")
            info2.pack()
        #if no flower is predicited
        else:
            label = tk.Label(root, text="NOT RECOGNIZED.TRY ANOTHER IMAGE!!!!", font=("Arial", 25))
            label.pack()

#Backend code for Clear label button 
def clear_fun():
    #delete labels
    label.pack_forget()
    info1.pack_forget()
    info2.pack_forget()
    my_img_label.pack_forget()


#MAIN -> Program is executed starting line 242
cnn = train_model()

#Create tkinter Window, shape, and title 
root = tk.Tk()
root.title("ODDISH Flower Recognition")
root.geometry('1300x700')

#Display ODDISH Logo on tkinter window
img = Image.open("logo.ico")
resize_image = img.resize((200, 200))
logo_img = ImageTk.PhotoImage(resize_image)
logo_img_label = tk.Label(root, image=logo_img)
logo_img_label.pack()
logo_img_label.image = logo_img
logo_img_label.pack()


#Display Intro message on tkinter winddow 
intro_label = tk.Label(root,
                       text = "Welome to ODDISH. Please upload your flower image and we will tell you the name and some cool facts!",
                       font=("Arial", 25))
intro_label.pack()

#Build file upload button, backened code for button @getfile(cnn) function defintition 
tk.Button(root, text="upload", command= lambda: get_file(cnn)).pack()

#Build clear label button, backend code for button @clear_fun function definition
tk.Button(root, text="clear", command = clear_fun).pack()

root.mainloop()

























