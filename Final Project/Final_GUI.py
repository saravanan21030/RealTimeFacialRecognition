from tkinter import *
import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sqlite3
import time
import datetime

Excited, Happy, Neutral, Exhausted = [0,0,0,0] 



root = Tk()
root.title("Dashboard: Sentimental Analysis")

l1 = StringVar()
l2 = StringVar()
l3 = StringVar()
l4 = StringVar()

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


#database
conn = sqlite3.connect('my_db.db')
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS sentimentRecords(id INTEGER PRIMARY KEY, datestamp TEXT, excited REAL, happy REAL, neutral REAL, exhausted REAL)")

def dynamic_data_entry():

    unix = int(time.time())
    date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S'))

    c.execute("INSERT INTO sentimentRecords (datestamp, excited, happy, neutral, exhausted) VALUES ( ?,?, ?, ?, ?)",
          (date, Excited, Happy, Neutral, Exhausted))

    conn.commit()
    time.sleep(1)
    
def graph_data():
    c.execute('SELECT datestamp, excited, happy, neutral, exhausted FROM sentimentRecords')
    data = c.fetchall()

    dates = []
    excited_ = []
    happy_ = []
    neutral_ = []
    exhausted_ = []
    
    for row in data:
        dates.append(parser.parse(row[0]))
        excited_.append(row[1])
        happy_.append(row[2])
        neutral_.append(row[3])
        exhausted_.append(row[4])

    figure = plt.Figure(figsize=(5,5), dpi=100)
    ax = figure.add_subplot(111)
    chart_type = FigureCanvasTkAgg(figure, root)
    chart_type.get_tk_widget().grid(row = 3, column = 0)
    ax.plot_date(dates,excited_,'-b')
    ax.plot_date(dates,happy_,'-g')
    ax.plot_date(dates,neutral_,'-r')
    ax.plot_date(dates,exhausted_,'-o')
    ax.set_title('Plot')
    #plt.show()
    
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def load():
    print("my function")
    cap=cv2.VideoCapture(0)
     
    global Excited, Happy, Neutral, Exhausted  
    
    while True:
        print("loop")
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
    
        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
    
            predictions = model.predict(img_pixels)
    
            #find max indexed array
            max_index = np.argmax(predictions[0])
    
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            
            
            
            if predicted_emotion == 'surprise':
                Excited +=1
            if predicted_emotion == 'happy':
                Happy +=1
            if predicted_emotion == 'neutral':
                Neutral +=1
            if predicted_emotion == 'sad':
                Exhausted +=1
                
            #print('Excited = ,', Excited, 'Happy =,', Happy, ',Neutral =,', Neutral, 'Exhausted = ,', Exhausted) 
            
            l1.set("Excited = " + str(Excited))
            l2.set("Happy = " + str(Happy))
            l3.set("Neutral = " + str(Neutral))
            l4.set("Exhausted = " + str(Exhausted))
            
            root.update()
            
            dynamic_data_entry()
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        #resized_img = cv2.resize(test_img, (1000, 700))
        #cv2.imshow('Facial emotion analysis ',resized_img)
    
        cv2.waitKey(100)
    
        #if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
         #   break
    
    cap.release()
    #cv2.destroyAllWindows




# Image Labels
im_excited = PhotoImage(file="Images/excited_1.gif")
label_excited = Label(root, image=im_excited).grid(row = 0, column = 0)

im_happy = PhotoImage(file="Images/happy_1.gif")
label_happy = Label(root, image=im_happy).grid(row = 0, column = 1)

im_neutral = PhotoImage(file="Images/neutral_1.gif")
label_neutral = Label(root, image=im_neutral).grid(row = 0, column = 2)

im_exhausted = PhotoImage(file="Images/sad_1.gif")
label_echausted = Label(root, image=im_exhausted).grid(row = 0, column = 3)


l1.set("Excited = 0")
l2.set("Happy = 0")
l3.set("Neutral = 0")
l4.set("Exhausted = 0")

my_label_1 = Label(root, textvariable = l1)
my_label_2 = Label(root, textvariable = l2)
my_label_3 = Label(root, textvariable = l3)
my_label_4 = Label(root, textvariable = l4)

my_label_1.grid(row = 1, column = 0)
my_label_2.grid(row = 1, column = 1)
my_label_3.grid(row = 1, column = 2)
my_label_4.grid(row = 1, column = 3)

button_start = Button(root, text = "START", command = load)
button_start.grid(row = 2, column = 0)

button_plot = Button(root, text = "PLOT", command = graph_data)
button_plot.grid(row = 2, column = 1)

root.mainloop()