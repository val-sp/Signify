import numpy as np
import cv2
import operator
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
import json
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

# Custom objects dictionary
custom_objects_dict = {
    'Sequential': Sequential,
    'InputLayer': InputLayer,
    'Conv2D': Conv2D,
    'MaxPooling2D': MaxPooling2D,
    'Flatten': Flatten,
    'Dense': Dense,
    'Dropout': Dropout
}

def load_model_safe(json_path, weights_path):
    """
    Robustly loads a Keras 2 model in Keras 3 by stripping ALL input shape 
    definitions from the JSON (both InputLayer and the first Conv2D layer).
    """
    with open(json_path, "r") as f:
        model_config = json.load(f)

    if 'config' in model_config and 'layers' in model_config['config']:
        layers = model_config['config']['layers']
        
        # 1. Remove the explicit InputLayer if it exists
        if layers and layers[0]['class_name'] == 'InputLayer':
            layers.pop(0)
        
        # 2. CRITICAL FIX: The first actual layer (Conv2D) likely still has 
        # 'batch_input_shape' inside its config. We must delete it.
        if layers:
            first_layer_config = layers[0]['config']
            # Remove keys that confuse Keras 3
            keys_to_remove = ['batch_input_shape', 'input_shape', 'batch_shape']
            for key in keys_to_remove:
                if key in first_layer_config:
                    del first_layer_config[key]

    # Convert the cleaned config back to string
    model_json_str = json.dumps(model_config)
    
    # Load the model structure (now it is "unbuilt" and purely sequential)
    model = model_from_json(model_json_str, custom_objects=custom_objects_dict)
    
    # 3. Manually build the model with the correct shape
    # We explicitly tell it: Batch size=None, H=128, W=128, Channels=1
    model.build(input_shape=(None, 128, 128, 1))
    
    # Load weights
    model.load_weights(weights_path)
    return model


# --- GLOBAL LOAD (Main Model) ---
loaded_model = load_model_safe("Models/model_last.json", "Models/model_last.weights.h5")


class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # --- LOAD MAIN MODEL ---
        self.loaded_model = load_model_safe("Models/model_last.json", "Models/model_last.weights.h5")

        # --- LOAD SUB-MODELS ---
        self.loaded_model_dru = load_model_safe("Models/model-bw_dru.json", "Models/model-bw_dru.h5")
        self.loaded_model_tkdi = load_model_safe("Models/model-bw_tkdi.json", "Models/model-bw_tkdi.h5")
        self.loaded_model_smn = load_model_safe("Models/model-bw_smn.json", "Models/model-bw_smn.h5")

        self.ct = {key: 0 for key in ascii_uppercase}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.current_symbol = "Empty"

        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("700x700")

        self.panel = tk.Label(self.root)
        self.panel.pack(padx=10, pady=10)

        self.panel2 = tk.Label(self.root)
        self.panel2.pack(padx=10, pady=10)

        self.symbol_label = tk.Label(self.root, text="Current Symbol: Empty", font=("Courier", 24))
        self.symbol_label.pack(pady=10)

        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            frame = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)
            imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk2
            self.panel2.config(image=imgtk2)

            self.symbol_label.config(text=f"Current Symbol: {self.current_symbol}")

        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))

        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))

        prediction = {'blank': result[0][0]}
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'R', 'U']:
            sub_pred = {
                'D': result_dru[0][0],
                'R': result_dru[0][1],
                'U': result_dru[0][2]
            }
            sub_pred = sorted(sub_pred.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = sub_pred[0][0]

        if self.current_symbol in ['D', 'I', 'K', 'T']:
            sub_pred = {
                'D': result_tkdi[0][0],
                'I': result_tkdi[0][1],
                'K': result_tkdi[0][2],
                'T': result_tkdi[0][3]
            }
            sub_pred = sorted(sub_pred.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = sub_pred[0][0]

        if self.current_symbol in ['M', 'N', 'S']:
            sub_pred = {
                'M': result_smn[0][0],
                'N': result_smn[0][1],
                'S': result_smn[0][2]
            }
            sub_pred = sorted(sub_pred.items(), key=operator.itemgetter(1), reverse=True)
            if sub_pred[0][0] == 'S':
                self.current_symbol = sub_pred[0][0]

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")
(Application()).root.mainloop()