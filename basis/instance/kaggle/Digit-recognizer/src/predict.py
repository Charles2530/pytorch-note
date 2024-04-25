import torch
import pandas as pd
import numpy as np
from main import CNNModel
# prediction
model_state_dict = torch.load(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Digit-recognizer/model/cnn_model.pkl')
model = CNNModel()
model.load_state_dict(model_state_dict)
model.eval()
# print prediction result to csv
test = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Digit-recognizer/data/test.csv', dtype=np.float32)
test = test.values/255
test = torch.from_numpy(test)
test = test.view(-1, 1, 28, 28)
output = model(test)
_, predicted = torch.max(output, 1)
predicted = predicted.numpy()
image_id = np.arange(1, len(predicted)+1)
submission = pd.DataFrame({'ImageId': image_id, 'Label': predicted})
submission.to_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Digit-recognizer/data/submission.csv', index=False)
