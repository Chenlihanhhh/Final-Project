import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class DiabetesEyeClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetes Eye Classifier")
        self.setGeometry(100, 100, 400, 300)

        self.model = None
        self.image_path = None

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)

        self.select_image_button = QPushButton("Select Image")
        self.select_image_button.clicked.connect(self.select_image)

        self.classify_button = QPushButton("Classify")
        self.classify_button.clicked.connect(self.classify_image)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.load_model_button)
        layout.addWidget(self.select_image_button)
        layout.addWidget(self.classify_button)
        layout.addWidget(self.image_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Model Files (*.h5)")
        if model_path:
            self.model = load_model(model_path)
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")

    def select_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def classify_image(self):
        if not self.model:
            QMessageBox.warning(self, "Error", "Please load a model first.")
            return
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image first.")
            return

        img = image.load_img(self.image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        prediction = self.model.predict(img_array)
        if prediction[0][0] > 0.5:
            result = "Positive (Diabetic Retinopathy)"
        else:
            result = "Negative (No Diabetic Retinopathy)"

        QMessageBox.information(self, "Prediction", f"Prediction: {result}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiabetesEyeClassifierApp()
    window.show()
    sys.exit(app.exec_())
