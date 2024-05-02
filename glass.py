# Team Members: Omar, Sam, Martin

from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split
from neural_net_UCI_data import parse_line, normalize

print("yo")


with open("glass_data.txt", "r") as f:
        training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# print(training_data)
td = normalize(training_data)
# print(td)

train, test = train_test_split(td)

nn = NeuralNet(9, 3, 1)

print(f"Print Train \n {train}\n")
print(f"Print Test \n {test}\n")

nn.train(train, iters=1000, print_interval=100, learning_rate=0.2)

for i in nn.test_with_expected(test):
    difference = round(abs(i[1][0] - i[2][0]), 3)
    print(f"desired: {i[1]}, actual: {i[2]} diff: {difference}")

# print(parse_line("1,1.52101,13.64,4.49,1.10,71.78,0.06,8.75,0.00,0.00,1")