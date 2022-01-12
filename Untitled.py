import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, VideoProcessorBase
import streamlit.components.v1 as components

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import scipy
import av

def main():
    class VideoProcessor():
        def __init__(self):
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.x_count = 0
            self.x_caught = 0
            self.f_count = 0
            self.model = tf.keras.models.load_model('./jab')
            self.tflite = tf.lite.Interpreter(model_path="model.tflite")
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                interpreter = self.tflite
                interpreter.allocate_tensors()

                edges = {
                    (0, 1): 'm',
                    (0, 2): 'c',
                    (1, 3): 'm',
                    (2, 4): 'c',
                    (0, 5): 'm',
                    (0, 6): 'c',
                    (5, 7): 'm',
                    (7, 9): 'm',
                    (6, 8): 'c',
                    (8, 10): 'c',
                    (5, 6): 'y',
                    (5, 11): 'm',
                    (6, 12): 'c',
                    (11, 12): 'y',
                    (11, 13): 'm',
                    (13, 15): 'm',
                    (12, 14): 'c',
                    (14, 16): 'c'
                }

                def draw_keypoints(frame, keypoints, confidence_threshold):
                    y, x, z = frame.shape
                    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

                    for kp in shaped:
                        ky, kx, kp_conf = kp
                        if kp_conf > confidence_threshold:
                            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

                def draw_connections(frame, keypoints, Edges, confidence_threshold):
                    y, x, c = frame.shape
                    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

                    for edge, color in Edges.items():
                        p1, p2 = edge
                        y1, x1, c1 = shaped[p1]
                        y2, x2, c2, = shaped[p2]

                        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)



                img = frame.to_ndarray(format="bgr24")

                img1 = img.copy() #copies current frame on camera
                img1 = tf.image.resize_with_pad(np.expand_dims(img1, axis=0),256,256) #resizes said image
                input_image= tf.cast(img1, dtype=tf.uint8) #casts said image as a float as required by the model

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                interpreter.set_tensor(input_details[0]["index"], input_image)
                interpreter.invoke()
                keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

                draw_keypoints(img, keypoints_with_scores, 0.2)
                draw_connections(img, keypoints_with_scores, edges, 0.2)

                img2 = img.copy()
                img2 = tf.image.resize_with_pad(np.expand_dims(img2, axis=0),256,256) 
                pred = self.model.predict(img2)

                self.f_count += 1

                cv2.putText(
                        img, f'{self.x_count}', (500, 100), self.font, 2, (0, 255, 0), 2)

                if pred[0][0]<pred[0][1]:
                    cv2.line(img, (150, 50), (75, 125), (0, 255, 0), 10)
                    cv2.line(img, (50, 100), (75, 125), (0, 255, 0), 10)
                    cv2.rectangle(img, (40, 40), (160, 160), (0, 0, 0), 5)
                    if self.f_count-5 > self.x_caught:
                        self.x_count+=1
                        self.x_caught = self.f_count

                else:
                    cv2.line(img, (50, 50), (150, 150), (0, 0, 255), 10)
                    cv2.line(img, (150, 50), (50, 150), (0, 0, 255), 10)
                    cv2.rectangle(img, (40, 40), (160, 160), (0, 0, 0), 5)

                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except:
                pass
            
 



    st.markdown("""
    <body style="background-color:#000000;">
    <h1 style='text-align: center; color: white;'>Welcome to the Online Dojo</h1>
    </body>

    """, unsafe_allow_html=True)

    st.markdown(
    """
    <style>
    .reportview-container {
        background: url("http://clubsolutionsmagazine.com/wp-content/uploads/2012/07/UFC-BJ-Bag-Room.jpg")
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown("""
    <body style="background-color:#000000;">
    <h1 style='text-align: center; color: red;'>NOW is the time to work...</h1>
    <h1 style='text-align: center; color: white;'>Not When you're in the ring...</h1>
    </body>

    """, unsafe_allow_html=True)


    if st.checkbox("Practice your jab"):
        webrtc_streamer(key="example", video_processor_factory=VideoProcessor)


if __name__ == "__main__":
    main()
