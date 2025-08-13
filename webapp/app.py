from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
try:
    data = joblib.load("credit_model.pkl")
    model = data["model"]
    feature_encoders = data["feature_encoders"]
    target_encoder = data["target_encoder"]
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        return handle_prediction()
    return render_template("home.html")

def handle_prediction():
    try:
        form_data = request.form.to_dict()
        processed_data = []
        
        features = [f for f in model.feature_names_in_ if f not in ("Unnamed: 0", "Risk")]
        
        for feature in features:
            value = form_data.get(feature)
            
            if not value:
                return render_template("home.html", error=f"Missing {feature}")
                
            if feature in feature_encoders:
                processed_data.append(feature_encoders[feature].transform([value])[0])
            else:
                processed_data.append(float(value))
        
        prediction = target_encoder.inverse_transform(
            model.predict(np.array([processed_data]))
        )[0]
        
        return render_template("result.html", prediction=prediction)
        
    except Exception as e:
        return render_template("home.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)