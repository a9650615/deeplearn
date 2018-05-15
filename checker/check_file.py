import re
from ast import literal_eval
import numpy as np

# with open('deep_data/network_layer_conv3_weights:0.txt', 'r') as myfile:
#     data=myfile.read().replace('\n', '')
#     #
#     a = data
#     a = re.sub('[^\r[]\s{1,}', ',', a)
#     print(a)
#     a = np.array(literal_eval(a))

data = np.loadtxt('deep_data/network_layer_conv3_weights:0.txt', unpack=True)

print(data)