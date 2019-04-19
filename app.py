# -*- coding: utf-8 -*-
from flask import Flask,request,render_template
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Model
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
import os
import pathlib
import warnings
warnings.filterwarnings("ignore")

# from waitress import serve

print('model_loading...')
json_file = open(r"model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(r"./model.h5") 
print('model loaded!!!')

graph = tf.get_default_graph()


app = Flask(__name__)




@app.route('/')
def index():
    return render_template("index.html")




temp = ['Non-Pneumonia','Pneumonia']

@app.route('/predict', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(file.filename)
    file.save(f)
    if pathlib.Path(r'static/img/chrt.jpeg').exists():
        os.remove(r'static/img/chrt.jpeg')
    img = mpimg.imread(file.filename)
    
    
#    fig = Figure()
#    fig, ax = plt.subplots(2,3,figsize=(15,7), sharex=True,sharey=True)
#    ax[0,0].imshow(img[:,:,0],cmap='hot')
#    ax[0,1].imshow(img[:,:,0],cmap='brg_r')
#    ax[0,2].imshow(img[:,:,0],cmap='gist_ncar')
#    ax[1,0].imshow(img[:,:,0],cmap='hsv_r')
#    ax[1,1].imshow(img[:,:,0],cmap='nipy_spectral')
#    ax[1,2].imshow(img[:,:,0],cmap='jet')
#    
   
    
    x = cv2.imread(file.filename)
    x = cv2.resize(x, (224,224))
    x = x.astype(np.float32)/255.
    x = x.reshape(1,224,224,3)
    
   
    
#    
#
    layer_outputs = [layer.output for layer in loaded_model.layers[:19]]
    
    

    activations=[]
    for i in range(1,len(layer_outputs)):
        activation_model = Model(inputs=loaded_model.input,outputs=layer_outputs[i])
        with graph.as_default():
            activations.append(activation_model.predict(x))
        

    fig, ax = plt.subplots(3, 6, figsize=(15,7), )
    activation_index=0
    for row in range(0,3):
        for col in range(0,6):
            ax[row][col].imshow(activations[activation_index][0, :, :, 0], cmap='hot')
            ax[row][col].axis('off')
            ax[row][col].set_title(layer_outputs[activation_index+1].name)
            activation_index += 1
    
    plt.tight_layout()
    plt.savefig('static/img/chrt.jpeg')
#    
##   # perform the prediction
    with graph.as_default():
        out = loaded_model.predict(x)
    response = np.argmax(out, axis=-1)
##    
#    
    return render_template('predict.html',pre = temp[response[0]])
##    






    


#    


#    # perform the prediction
#    out = loaded_model.predict(x)
#    response = np.argmax(out, axis=-1)
#    return str(response[0])





if __name__ == "__main__":
   app.run('0.0.0.0',port = 5000,threaded=False,debug=True)
#     serve(app, host='0.0.0.0', port=5000)
    
