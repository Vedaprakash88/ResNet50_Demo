import os
from .classifier import ResNetClassifier

class ResNetOrchestrator:
    """
    Orchestrates the entire image classification training pipeline.
    """
    def __init__(self, config):
        self.config = config

    def run_pipeline(self, run_training=True):
        """
        Runs the model preparation, training, evaluation, and serialization steps.
        """
        print("\n" + "="*80)
        print("🚀 RESNET50 TRANSFER LEARNING ORCHESTRATOR STARTED")
        print("="*80 + "\n")

        if not run_training:
            print("Training is disabled. Skipping pipeline execution.")
            return

        train_dir = self.config.get('train_dir')
        val_dir = self.config.get('val_dir')
        model_dir = self.config.get('model_dir')
        log_dir = self.config.get('log_dir')
        checkpoint_dir = self.config.get('checkpoint_dir')

        if not all([train_dir, val_dir, model_dir, log_dir, checkpoint_dir]):
            print("❌ Pipeline execution aborted: Missing directories in config.")
            print(f"  train_dir: {train_dir}")
            print(f"  val_dir: {val_dir}")
            print(f"  model_dir: {model_dir}")
            return

        if not os.path.exists(train_dir):
            print(f"❌ Pipeline execution aborted: Training directory '{train_dir}' does not exist.")
            return
        if not os.path.exists(val_dir):
            print(f"❌ Pipeline execution aborted: Validation directory '{val_dir}' does not exist.")
            return

        print(f"Configured Training Directories:")
        print(f"  - Train Directory:      {train_dir}")
        print(f"  - Validation Directory: {val_dir}")
        print(f"  - Model Save Directory: {model_dir}")
        print(f"  - Log Save Directory:   {log_dir}")
        print(f"  - Checkpoint Directory: {checkpoint_dir}")
        print(f"Hyperparameters:")
        print(f"  - Batch Size:    {self.config.get('batch_size')}")
        print(f"  - Epochs:        {self.config.get('epochs')}")
        print(f"  - Learning Rate: {self.config.get('learning_rate')}")
        print(f"  - Early Stop Patience: {self.config.get('patience')}")

        # Instantiate classifier
        classifier = ResNetClassifier(
            train_dir=train_dir,
            val_dir=val_dir,
            model_dir=model_dir,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            batch_size=self.config.get('batch_size', 32),
            epochs=self.config.get('epochs', 25),
            learning_rate=self.config.get('learning_rate', 0.0001),
            patience=self.config.get('patience', 4)
        )

        # 1. Prepare data
        classifier.prepare_data(visualize_samples=True)

        # 2. Build model
        classifier.build_model()

        # 3. Train
        classifier.train(save_plots=True)

        # 4. Evaluate
        classifier.evaluate()

        # 5. Save model
        model_name = self.config.get('model_name', 'brain_tumor_resnet50.keras')
        classifier.save_model(model_name)

        print("\n" + "="*80)
        print("🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
