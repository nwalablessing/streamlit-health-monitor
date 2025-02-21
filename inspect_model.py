import pickle
import os

# Path to your .pkl file
file_path = r"C:\Users\nwala\Documents\Spring Final Year project 2025\sepsis_best_model.pkl"

try:
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found at: {file_path}")
    else:
        # Load the .pkl file
        with open(file_path, "rb") as f:
            loaded_object = pickle.load(f)
        
        # Check the type of the loaded object
        print(f"Loaded object type: {type(loaded_object)}")
        
        # Check if it's a machine learning model
        if hasattr(loaded_object, "predict"):
            print("The .pkl file contains a trained model.")
            
            # Check if the model has feature names (if it is scikit-learn or similar)
            if hasattr(loaded_object, "feature_names_in_"):
                print("Feature Names:", loaded_object.feature_names_in_)
                print(f"Number of Features: {len(loaded_object.feature_names_in_)}")
            else:
                print("The model does not explicitly provide feature names.")
            
            # If the model expects a specific number of features
            if hasattr(loaded_object, "n_features_in_"):
                print(f"Model expects {loaded_object.n_features_in_} features.")
        else:
            print("The .pkl file does not contain a machine learning model.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
