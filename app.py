import streamlit as st

from keras.models import load_model # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def main() :
    st.title("깨끗한 방인지 더러운 방인지")
    st.info("방 사진을 업로드하면, 깨끗한 방인지 더러운 방인지 알려드립니다.")
    
    # 파일 업로더
    image = st.file_uploader("이미지 파일을 업로드 하세요", ["jpg", "png", "jpeg"])
    
    if image is not None :
        st.image(image)

        image = Image.open(image)

        # Load the model
        model = load_model("model/keras_model.h5", compile=False)

        # Load the labels
        class_names = open("model/labels.txt", "r", encoding= "utf-8").readlines()

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

        st.info(f"이 방은 {class_name[2:]} 방입니다. 확률은 {confidence_score:.2f} 입니다")
        



if __name__ == "__main__" :
    main()