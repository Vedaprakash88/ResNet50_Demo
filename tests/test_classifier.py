import os
import unittest
import tempfile
import shutil
import cv2
import numpy as np
from resnet_classifier import ResNetClassifier

def create_dummy_jpeg(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Generate 256x256 random image
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    cv2.imwrite(path, img)

class TestResNetClassifier(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.train_dir = os.path.join(self.temp_dir, "train")
        self.val_dir = os.path.join(self.temp_dir, "val")
        self.model_dir = os.path.join(self.temp_dir, "model")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")

        # Create dummy images (4 images per class for 2 classes)
        self.classes = ["glioma", "meningioma"]
        for cls in self.classes:
            for i in range(4):
                img_train_path = os.path.join(self.train_dir, cls, f"img_{i}.jpeg")
                img_val_path = os.path.join(self.val_dir, cls, f"img_{i}.jpeg")
                create_dummy_jpeg(img_train_path)
                create_dummy_jpeg(img_val_path)

        self.classifier = ResNetClassifier(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            model_dir=self.model_dir,
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            batch_size=2,
            epochs=1,
            weights=None # Run tests offline and fast
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_classifier_pipeline(self):
        # 1. Prepare Data
        self.classifier.prepare_data(visualize_samples=True)
        self.assertEqual(len(self.classifier.class_names), 2)
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "dataset_samples.png")))

        # 2. Build Model
        self.classifier.build_model()
        self.assertIsNotNone(self.classifier.model)

        # 3. Train
        history = self.classifier.train(save_plots=True)
        self.assertIsNotNone(history)
        
        # Verify plots were saved
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "resnet_loss_curves.png")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "resnet_accuracy_curves.png")))

        # 4. Evaluate
        scores = self.classifier.evaluate()
        self.assertIsNotNone(scores)

        # 5. Save model
        model_name = "test_resnet.keras"
        model_path = self.classifier.save_model(model_name)
        self.assertTrue(os.path.exists(model_path))

if __name__ == '__main__':
    unittest.main()
