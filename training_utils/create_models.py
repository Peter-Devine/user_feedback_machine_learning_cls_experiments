from data_utils.create_features import get_features
from training_utils.deep_training import train_deep_model
from training_utils.classical_training import train_classical_model

def create_models(model_config):
    print(f"Now training all models")
    
    print("Reading data...")
    train_dataset, eval_dataset, test_dataset, vocab_size = get_features(model_config)
        
    print("Training...")
    train_deep_model(model_config, train_dataset, eval_dataset, vocab_size)
    