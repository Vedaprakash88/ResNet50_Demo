import os
import configparser

def load_config(config_path=None):
    """
    Loads configurations from a .ini file and formats them as a pipeline config dictionary.
    """
    if config_path is None:
        # Resolve config.ini relative to current working directory or package root
        possible_paths = [
            "config.local.ini",
            os.path.join(os.getcwd(), "config.local.ini"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.local.ini"),
            "config.ini",
            os.path.join(os.getcwd(), "config.ini"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.ini")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        if not config_path:
            raise FileNotFoundError("Could not locate config.ini or config.local.ini. Please supply a direct path.")

    print(f"Reading configuration from: {config_path}")
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Resolve paths
    train_dir = parser.get('paths', 'train_dir', fallback='')
    val_dir = parser.get('paths', 'val_dir', fallback='')
    out_dir = parser.get('paths', 'output_dir', fallback='')

    # Populate config dictionary
    config = {
        'train_dir': train_dir,
        'val_dir': val_dir,
        'output_dir': out_dir,
        'log_dir': os.path.join(out_dir, 'call_logs') if out_dir else '',
        'checkpoint_dir': os.path.join(out_dir, 'chk_pt') if out_dir else '',
        'model_dir': os.path.join(out_dir, 'my_model') if out_dir else '',
        
        'batch_size': parser.getint('training', 'batch_size', fallback=32),
        'epochs': parser.getint('training', 'epochs', fallback=25),
        'learning_rate': parser.getfloat('training', 'learning_rate', fallback=0.0001),
        'model_name': parser.get('training', 'model_name', fallback='brain_tumor_resnet50.keras'),
        'patience': parser.getint('training', 'patience', fallback=4),
    }

    return config
