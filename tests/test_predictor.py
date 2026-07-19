import os
import unittest
import tempfile
import shutil
from resnet_classifier import ResNetClassifier, ResNetPredictor
from tests.test_classifier import create_dummy_jpeg

class TestResNetPredictor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.train_dir = os.path.join(self.temp_dir, "train")
        self.val_dir = os.path.join(self.temp_dir, "val")
        self.model_dir = os.path.join(self.temp_dir, "model")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")

        # Create dummy images
        self.classes = ["glioma", "meningioma"]
        for cls in self.classes:
            for i in range(4):
                create_dummy_jpeg(os.path.join(self.train_dir, cls, f"img_{i}.jpeg"))
                create_dummy_jpeg(os.path.join(self.val_dir, cls, f"img_{i}.jpeg"))

        # Train a dummy model to save it
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
        self.classifier.prepare_data(visualize_samples=False)
        self.classifier.build_model()
        self.classifier.train(save_plots=False)
        
        self.model_name = "test_resnet_predict.keras"
        self.model_path = self.classifier.save_model(self.model_name)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_predictor_inference(self):
        # 1. Instantiate predictor
        predictor = ResNetPredictor(model_path=self.model_path)
        self.assertIsNotNone(predictor.model)

        # 2. Run inference on a dummy image
        test_img_path = os.path.join(self.temp_dir, "test_mri.jpeg")
        create_dummy_jpeg(test_img_path)

        yhat = predictor.predict(test_img_path)
        self.assertEqual(yhat.shape, (1, 2))

        # 3. Test class prediction helper
        idx, name, probs = predictor.get_predicted_class(test_img_path, class_names=self.classes)
        self.assertIn(idx, [0, 1])
        self.assertEqual(name, self.classes[idx])
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
