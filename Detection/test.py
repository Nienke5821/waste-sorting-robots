from ultralytics import YOLO
import json

if __name__ == "__main__":

    def evalutate_models(model_paths):
        """Evaluates multiple YOLO-models on a test dataset found by the data_config, and returns key performance metrics
       
        Args:
        model_paths (list of str): Difference in Paths to best.pt file from all models who will be evaluated

        Returns:
        list of list of float: A list where each element contains [mAP@0.5, mean recall] for each model in model_paths
        """ 
        model_results = []
        for i in range(len(model_paths)):
            current_model = model_paths[i]
            model_path = f"C:/...../models/{current_model}/best.pt"
            print("testing model:", current_model)

            # Load model
            model = YOLO(model_path)  
            print("model found")

            # Run validation on test dataset
            data_config = "C:/...../config_test.yaml" 
            results = model.val(data=data_config)

            # Collect resulst and save
            combined_results = []        
            combined_results.append(results.box.map50) # Mean average precision at IoU=0.50
            combined_results.append(results.box.mr) # Mean recall
            results_float = [float(x) for x in combined_results]
            print(results_float)
            model_results.append(results_float)
        return model_results
    
    
    model_paths = ["A/best_after_warp", "B/erasing_0_6", "B/erasing_0_8", "B/hsv_h_0_1", "B/hsv_h_0_2", "B/hsv_v_0_1", "B/hsv_v_0_8"]
    results = evalutate_models(model_paths)
    

    # Save results to JSON file
    with open("model_results.json", "w") as f:
        json.dump({
            "model_paths": model_paths,
            "results": results
        }, f)



    



