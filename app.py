import joblib
import gradio as gr

# Load the saved model
model = joblib.load("iris_model.pkl")

# Class labels
classes = ["Setosa", "Versicolor", "Virginica"]

# Prediction function
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
    return classes[prediction]

# Create Gradio interface
iface = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)")
    ],
    outputs=gr.Textbox(label="Predicted Species"),
    title="Iris Flower Predictor",
    description="Enter the measurements of the Iris flower and get the predicted species."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()

