import io
import numpy as np
from PIL import Image
import tensorflow as tf
from google.colab import files

# Load the model
model = tf.keras.models.load_model('tumor_classification.h5')

def preprocess_image(uploaded_image):
    resized_image = uploaded_image.resize((224, 224))
    image_array = np.array(resized_image)
    return image_array

def prediction(image_array):
    pred = model.predict(np.expand_dims(image_array, axis=0))
    return pred

def main():
    uploaded = files.upload()
    for fn in uploaded.keys():
        image = Image.open(io.BytesIO(uploaded[fn]))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        image_array = preprocess_image(image)
        ans = prediction(image_array)
        classes = ['No Tumor',
                   'Tumor']
        print("Prediction Probabilities:")
        for i in range(len(classes)):
            print(f"Class Name:{classes[i]} ({ans[0][i]})")

        result = 'Predict Result : {}'.format(classes[np.argmax(ans)])
        print(result)

if __name__ == "__main__":
    main()
