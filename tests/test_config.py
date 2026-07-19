import os
import unittest
import tempfile
import shutil
from resnet_classifier import load_config

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ini_path = os.path.join(self.temp_dir, "test_config.ini")
        
        ini_content = """[paths]
train_dir = /dummy/train
val_dir = /dummy/val
output_dir = /dummy/output

[training]
batch_size = 16
epochs = 5
learning_rate = 0.0005
model_name = dummy_model.keras
patience = 2
"""
        with open(self.ini_path, 'w') as f:
            f.write(ini_content)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_config(self):
        config = load_config(self.ini_path)
        self.assertEqual(config['train_dir'], "/dummy/train")
        self.assertEqual(config['val_dir'], "/dummy/val")
        self.assertEqual(config['output_dir'], "/dummy/output")
        self.assertEqual(config['batch_size'], 16)
        self.assertEqual(config['epochs'], 5)
        self.assertEqual(config['learning_rate'], 0.0005)
        self.assertEqual(config['model_name'], "dummy_model.keras")
        self.assertEqual(config['patience'], 2)

if __name__ == '__main__':
    unittest.main()
