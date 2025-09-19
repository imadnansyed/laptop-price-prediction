import os
import pandas as pd
import pickle

from src.utils.predict_utils import calculate_ppi

PREPROCESSOR_PATH = os.path.join("artifacts", "pipelines", "best_preprocessor_pipeline.pkl")
PREDICT_PATH  = os.path.join("artifacts", "pipelines", "prediction_pipeline.pkl")

print (f"Loading preprocessor from: {PREPROCESSOR_PATH}")
print (f"Loading prediction model from: {PREDICT_PATH}")
preprocessor = pickle.load(open(PREPROCESSOR_PATH, 'rb'))
predict_pipeline = pickle.load(open(PREDICT_PATH, 'rb'))


def predict(input_data):
    print(preprocessor.get_feature_names_out())
    # processed_data = preprocessor.transform(input_data)
    print(predict_pipeline)
    preds = predict_pipeline.predict(input_data)
    return preds

class CustomData:
    def __init__(self, form):
        # Collect inputs
        self.company = form.get('company')
        self.typename = form.get('typename')
        self.ram = int(form.get('ram'))
        self.screen_size = float(form.get('screen_size'))
        self.resolution = form.get('resolution')
        self.cpu_brand = form.get('cpu_brand')
        self.ssd = int(form.get('ssd'))
        self.hdd = int(form.get('hdd'))
        self.gpu_name = form.get('gpu_name')
        self.os_type = form.get('os')
        
def get_data_as_dataframe(data:CustomData):
    
    data = {
        "company": data.company,
        "typename": data.typename,
        "ram": int(data.ram),
        "ppi": float(calculate_ppi(data.resolution,float(data.screen_size))),
        "cpu brand": data.cpu_brand,
        "ssd": int(data.ssd),
        "hdd": int(data.hdd),
        "gpu_name": data.gpu_name,
        "os": data.os_type
    }
    
    df = pd.DataFrame([data])
    
    return df