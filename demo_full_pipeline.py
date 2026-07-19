import os
import sys
import subprocess

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure we run from the directory containing this script so relative paths are correct
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

# Check, install requirements, and import
try:
    from resnet_classifier import load_config, ResNetOrchestrator, ResNetPredictor
except (ImportError, ModuleNotFoundError) as e:
    print(f"Missing dependencies or resnet_classifier package not installed ({e}).")
    print("Installing requirements from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Installing local resnet_classifier package in editable mode...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("All dependencies installed successfully!\n")
    except subprocess.CalledProcessError as install_err:
        print(f"Error during package installation: {install_err}")
        sys.exit(1)

    from resnet_classifier import load_config, ResNetOrchestrator, ResNetPredictor

def run_test():
    print("=" * 80)
    print("🎬 STARTING RESNET50 TRANSFER LEARNING DEMO & TEST")
    print("=" * 80)

    # 1. Load configuration (prioritizing config.local.ini)
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Check if directories exist
    train_dir = config.get('train_dir')
    val_dir = config.get('val_dir')
    if not train_dir or not val_dir or not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Error: Train or validation directory does not exist.")
        print(f"  train_dir: {train_dir}")
        print(f"  val_dir: {val_dir}")
        print("Please configure paths in config.local.ini first.")
        return

    # For quick testing, override epochs to 2
    config_copy = config.copy()
    config_copy['epochs'] = 2
    print(f"Overriding epochs to {config_copy['epochs']} for a quick run.")

    # 2. Run the pipeline orchestrator
    orchestrator = ResNetOrchestrator(config_copy)
    orchestrator.run_pipeline(run_training=True)

    # 3. Test on-the-fly Predictor Inference on validation image
    print("--- TESTING ON-THE-FLY INFERENCE ---")
    model_name = config_copy['model_name']
    model_dir = config_copy['model_dir']
    
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Search for an image file in the validation directory to run inference on
    found_image = None
    for root, _, files in os.walk(val_dir):
        for f in files:
            if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp')):
                found_image = os.path.join(root, f)
                break
        if found_image:
            break

    if not found_image:
        print("No validation image found in dataset to test inference.")
        return

    print(f"Running inference on: {found_image}")
    predictor = ResNetPredictor(
        model_path=model_path
    )

    # Automatically infer class names from validation directories
    class_names = sorted(os.listdir(val_dir))
    class_idx, class_name, probs = predictor.get_predicted_class(
        image_path=found_image,
        class_names=class_names
    )

    print("\n" + "=" * 40)
    print("🔮 PREDICTION RESULT")
    print("=" * 40)
    print(f"Image Path: {found_image}")
    print(f"Predicted Class Index: {class_idx}")
    print(f"Predicted Class Name:  {class_name}")
    print(f"Prediction Probabilities: {probs}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_test()
