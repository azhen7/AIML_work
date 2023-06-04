import onnxruntime as ort
import numpy as np
from  periodictable import elements

print(elements._element)

ort_sess = ort.InferenceSession('model_pytorch.onnx')

x = np.zeros((120), dtype=np.float32)
x[42] = 0.67
x[75] = 0.20
x[14] = 0.13

x = x.reshape((1, 1, 10, 12))*100

outputs = ort_sess.run(None, {'input.1': x})

print(outputs[0], outputs[1])
isconduc = False
pred_value = outputs[0]
if(outputs[1][0][1] > outputs[1][0][0]):
    isconduc = True
else:
    pred_value = 0   

print("test on Mo0.67Re0.20Si0.13")
print("ground:     supderconductive: yes, TC <= 6 ")
print("prediction: supderconductive:", isconduc,  "confidence_level: ", outputs[1][0][1],  "TC <= ", pred_value)


x = np.zeros((120), dtype=np.float32)
x[42] = 0.67
x[44] = 0.20
x[14] = 0.13
x = x.reshape((1, 1, 10, 12))*100

outputs = ort_sess.run(None, {'input.1': x})

#print(outputs[0], outputs[1])

isconduc = False
pred_value = outputs[0]
if(outputs[1][0][1] > outputs[1][0][0]):
    isconduc = True
else:
    pred_value = 0   

print("test on Mo0.67Ru0.20Si0.13")
print("ground:     supderconductive: false, TC <= 0 ")
print("prediction: supderconductive:", isconduc, "confidence_level: ", outputs[1][0][0],  "TC <= ", pred_value)


x = np.zeros((120), dtype=np.float32)
x[42] = 0.67
x[45] = 0.20
x[32] = 0.13
x = x.reshape((1, 1, 10, 12))*100

outputs = ort_sess.run(None, {'input.1': x})

isconduc = False
pred_value = outputs[0]
if(outputs[1][0][1] > outputs[1][0][0]):
    isconduc = True
else:
    pred_value = 0   
print("test on Mo0.67Rh0.20Ge0.13")
print("ground:     supderconductive: false, TC <= 0 ")
print("prediction: supderconductive:", isconduc, "confidence_level: ", outputs[1][0][0],  "TC <= ", pred_value)
