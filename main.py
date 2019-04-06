import csv
import numpy as np
from sklearn.neural_network import MLPClassifier

train_data=np.zeros((9*1500,4096))
train_labels=np.zeros((9*1500))
test_data=np.zeros((1500,4096))
test_labels=np.zeros((1500))

# with open('output.csv','w', newline='') as output_file:
#     prediction_writer=csv.writer(output_file)
#
#     for i in range(100):
#         prediction_writer.writerow([i+1])

mlp_classifier=MLPClassifier(
    hidden_layer_sizes=(30,25),
    activation='logistic',
    solver='sgd',
    alpha=0.001,
    batch_size=200,
    learning_rate='adaptive',
    learning_rate_init=0.003,
    power_t=0.5,#pt learning_rate=invscaling
    max_iter=1000,
    shuffle=True,
    random_state=None,
    tol=0.00003,
    momentum=0.7,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=7
)

loaded_labels=np.loadtxt('data/train_labels.csv', 'int')
loaded_samples=np.loadtxt('data/train_samples.csv', delimiter=',')

# print(train_samples.shape[0])
# print(train_data.shape)

j=0
k=0

for i in range(loaded_samples.shape[0]): #original
    if i%10:
        train_data[j]=loaded_samples[i]
        train_labels[j]=loaded_labels[i]
        j+=1
    else:
        test_data[k]=loaded_samples[i]
        test_labels[k]=loaded_labels[i]
        k+=1

mlp_classifier.fit(train_data,train_labels)

# mlp_classifier.fit(loaded_samples,loaded_labels)
#
# test_samples=np.loadtxt('data/test_samples.csv', delimiter=',')
# predicted_labels=mlp_classifier.predict(test_samples)
#
# with open('output.csv','w', newline='') as output_file:
#     prediction_writer=csv.writer(output_file)
#
#     for i in range(predicted_labels.shape[0]):
#         prediction_writer.writerow([i+1, predicted_labels[i]])


predicted_labels=mlp_classifier.predict(test_data)

print(predicted_labels)
j=0
for i in range(predicted_labels.shape[0]):
    if predicted_labels[i]==test_labels[i]:
        j+=1
print(j,j/1500)

confusion_matrix=np.zeros((8,8), dtype=np.int)
for i in range(predicted_labels.shape[0]):
    y=predicted_labels[i]
    x=test_labels[i]
    confusion_matrix[int(x)][int(y)]+=1
print(confusion_matrix)

confusion_matrix=np.zeros((8,8), dtype=np.int)
predicted_labels=mlp_classifier.predict(train_data)
j=0
for i in range(predicted_labels.shape[0]):
    if predicted_labels[i]==train_labels[i]:
        j+=1
print(j,j/13500)
for i in range(train_labels.shape[0]):
    y=predicted_labels[i]
    x=train_labels[i]
    confusion_matrix[int(x)][int(y)]+=1
print(confusion_matrix)



# stock:
# mlp_classifier=MLPClassifier(
#     hidden_layer_sizes=(20,20),
#     activation='relu',
#     solver='sgd',
#     alpha=0.001,
#     batch_size=200,
#     learning_rate='constant',
#     power_t=0.5,#pt learning_rate=invscaling
#     max_iter=200,
#     shuffle=True,
#     random_state=None,
#     tol=0.0001,
#     momentum=0.9,
#     early_stopping=False,
#     validation_fraction=0.1,
#     n_iter_no_change=10
# )

# mlp_classifier=MLPClassifier( 1452 0.972
#     hidden_layer_sizes=(30,25),
#     activation='logistic',
#     solver='sgd',
#     alpha=0.001,
#     batch_size=200,
#     learning_rate='adaptive',
#     learning_rate_init=0.003,
#     power_t=0.5,#pt learning_rate=invscaling
#     max_iter=1000,
#     shuffle=True,
#     random_state=None,
#     tol=0.00003,
#     momentum=0.7,
#     early_stopping=False,
#     validation_fraction=0.1,
#     n_iter_no_change=7
# )