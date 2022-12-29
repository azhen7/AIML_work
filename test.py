
import onnxruntime as ort
import numpy as np


ort_sess = ort.InferenceSession('./AIML_work/model_pytorch.onnx')

x = np.zeros((120), dtype=np.float32)
x[47] = 0
x[4] = 0.93
x[75] = 0.07

x = x.reshape((1, 1, 10, 12))*100

outputs = ort_sess.run(None, {'input.1': x})

print(outputs[0])

x = np.zeros((120), dtype=np.float32)
x[47] = 0.33
x[83] = 0.67

x = x.reshape((1, 1, 10, 12))*100

outputs = ort_sess.run(None, {'input.1': x})

print(outputs[0])


x = np.zeros((120), dtype=np.float32)
x[47] = 0.15
x[83] = 0.85

x = x.reshape((1, 1, 10, 12))*100

outputs = ort_sess.run(None, {'input.1': x})

print(outputs[0])