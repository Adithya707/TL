from flask import Flask,render_template,request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.models import load_model
import json

app = Flask(__name__)
#model = VGG16()
model =load_model('model.h5')

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    
    imagefile = request.files['imagefile']
    image_path = "./images/"+imagefile.filename
    imagefile.save(image_path)
    image = load_img(image_path,target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1],label[2]*100)
    return render_template('index.html',prediction=classification)

def decode_predictions(preds, top=4, class_list_path='index.json'):
    if len(preds.shape) != 2 or preds.shape[1] != 2: # your classes number
         raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
    index_list = json.load(open(class_list_path))
    results = []
    for pred in preds:
       top_indices = pred.argsort()[-top:][::-1]
       result = [tuple(index_list[str(i)]) + (pred[i],) for i in top_indices]
       result.sort(key=lambda x: x[2], reverse=True)
       results.append(result)
    return results




if __name__ == '__main__':
    app.run(port=3000, debug=False,threaded=False)