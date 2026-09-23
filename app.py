from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import pyttsx3

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        file = request.files["image"]

        # Read image
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # Resize
        img = cv2.resize(img, (64, 64))

        # Flatten
        img = img.flatten().reshape(1, -1)

        # Prediction
        prediction = model.predict(img)[0]

        result = f"₹{prediction} Rupees"

        # Voice output
        
        engine = pyttsx3.init()
        engine.say(f"This is {prediction} rupees")
        engine.runAndWait()
        engine.stop()
        

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=False)
