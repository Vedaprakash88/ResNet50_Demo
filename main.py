import os
import easygui
from resnet_classifier import load_config, ResNetOrchestrator

def main():
    print("="*80)
    print("🧠 BRAIN TUMOR RESNET50 CLASSIFIER ENTRYPOINT")
    print("="*80)

    # 1. Load config
    try:
        config = load_config()
    except Exception as e:
        print(f"Warning loading config: {e}. Fallback to manual GUI selections.")
        config = {}

    # 2. Check if folders are configured/exist, otherwise prompt user graphically
    train_dir = config.get('train_dir', '')
    if not train_dir or not os.path.exists(train_dir):
        train_dir = easygui.diropenbox(msg="Select training directory containing class subfolders", title="Select Train Directory")
        if not train_dir:
            print("Pipeline cancelled: No training directory selected.")
            return
        config['train_dir'] = train_dir

    val_dir = config.get('val_dir', '')
    if not val_dir or not os.path.exists(val_dir):
        val_dir = easygui.diropenbox(msg="Select validation/testing directory containing class subfolders", title="Select Validation Directory")
        if not val_dir:
            print("Pipeline cancelled: No validation directory selected.")
            return
        config['val_dir'] = val_dir

    out_dir = config.get('output_dir', '')
    if not out_dir:
        out_dir = easygui.diropenbox(msg="Select target directory to save output data (logs, models, checkpoints)", title="Select Output Location")
        if not out_dir:
            # Sibling to train_dir
            out_dir = os.path.join(os.path.dirname(os.path.abspath(train_dir.rstrip('/\\'))), "resnet_output")
        config['output_dir'] = out_dir

    # Refresh subfolder paths in config dict
    config['log_dir'] = os.path.join(out_dir, 'call_logs')
    config['checkpoint_dir'] = os.path.join(out_dir, 'chk_pt')
    config['model_dir'] = os.path.join(out_dir, 'my_model')

    print(f"Configured Train Root: {config['train_dir']}")
    print(f"Configured Validation Root: {config['val_dir']}")
    print(f"Configured Output Target: {config['output_dir']}")
    print(f"Model File Name: {config.get('model_name')}")

    # 3. Run pipeline
    orchestrator = ResNetOrchestrator(config)
    orchestrator.run_pipeline(run_training=True)

if __name__ == "__main__":
    main()
