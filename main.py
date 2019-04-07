import csv
import numpy as np
from sklearn.neural_network import MLPClassifier

train_data=np.zeros((9*1500,4096))
train_labels=np.zeros((9*1500))
test_data=np.zeros((1500,4096))
test_labels=np.zeros((1500))

mlp_classifier=MLPClassifier( #1465 0.998
    hidden_layer_sizes=([40,40]),
    activation='logistic',
    solver='sgd',
    alpha=0.001, #0.001 1463
    batch_size=200, #200: 1439, .984
    learning_rate='adaptive',
    learning_rate_init=0.5,
    power_t=0.5,#pt learning_rate=invscaling
    max_iter=850,
    shuffle=False,
    random_state=None,
    tol=0.0001,#+1 0
    momentum=0.7,
    early_stopping=False,
    validation_fraction=0.3,
    n_iter_no_change=10
)

loaded_labels=np.loadtxt('data/train_labels.csv', 'int')
loaded_samples=np.loadtxt('data/train_samples.csv', delimiter=',')

# print(train_samples.shape[0])
# print(train_data.shape)

# j=0
# k=0
#
# x=0#new
# rare_labels=np.zeros(0)#new
# rare_data=np.zeros((0,4096))#new
# for i in range(loaded_samples.shape[0]): #original
#     if i%10:
#         train_data[j]=loaded_samples[i]
#         train_labels[j]=loaded_labels[i]
#         j+=1
#     else:
#         test_data[k]=loaded_samples[i]
#         test_labels[k]=loaded_labels[i]
#         k+=1
#     if loaded_labels[i] in [5,6,7]:
#         rare_data=np.append(rare_data,loaded_samples[i])
#         rare_labels=np.append(rare_labels,loaded_labels[i])
#         x+=1
#
# rare_data=rare_data.reshape((int(rare_data.size/4096),4096))#new
# print(rare_data.shape)#new
# train_data=np.append(train_data,rare_data)#new
# train_labels=np.append(train_labels,rare_labels)#new
# #train_data=np.append(train_data,rare_data)#new
# #train_labels=np.append(train_labels,rare_labels)#new
# # train_data=np.append(train_data,rare_data)#new
# # train_labels=np.append(train_labels,rare_labels)#new
# train_data=train_data.reshape((13500+432*1,4096))
# print(train_data.shape)#new
# mlp_classifier.fit(train_data,train_labels)

x=0
rare_labels=np.zeros(0)
rare_data=np.zeros((0,4096))
for i in range(loaded_samples.shape[0]):
    if loaded_labels[i] in [5,6,7]:
        rare_data=np.append(rare_data,loaded_samples[i])
        rare_labels=np.append(rare_labels,loaded_labels[i])
        x+=1
rare_data=rare_data.reshape((int(rare_data.size/4096),4096))
print(rare_data.shape[0])
loaded_samples=np.append(loaded_samples,rare_data)
loaded_labels=np.append(loaded_labels,rare_labels)
#loaded_samples=np.append(loaded_samples,rare_data)
#loaded_labels=np.append(loaded_labels,rare_labels)
#loaded_samples=np.append(loaded_samples,rare_data)
#loaded_labels=np.append(loaded_labels,rare_labels)
loaded_samples=loaded_samples.reshape((15000+rare_data.shape[0]*1,4096))
mlp_classifier.fit(loaded_samples,loaded_labels)

test_samples=np.loadtxt('data/test_samples.csv', delimiter=',')
predicted_labels=mlp_classifier.predict(test_samples)

with open('output.csv','w', newline='') as output_file:
    prediction_writer=csv.writer(output_file)

    for i in range(predicted_labels.shape[0]):
        prediction_writer.writerow([i+1, predicted_labels[i]])


# predicted_labels=mlp_classifier.predict(test_data)
#
# print(predicted_labels)
# j=0
# for i in range(predicted_labels.shape[0]):
#     if predicted_labels[i]==test_labels[i]:
#         j+=1
# print(j,j/1500)
#
# confusion_matrix=np.zeros((8,8), dtype=np.int)
# for i in range(predicted_labels.shape[0]):
#     y=predicted_labels[i]
#     x=test_labels[i]
#     confusion_matrix[int(x)][int(y)]+=1
# print(confusion_matrix)
#
# confusion_matrix=np.zeros((8,8), dtype=np.int)
# predicted_labels=mlp_classifier.predict(train_data)
# j=0
# for i in range(predicted_labels.shape[0]):
#     if predicted_labels[i]==train_labels[i]:
#         j+=1
# print(j,j/predicted_labels.shape[0])
# for i in range(train_labels.shape[0]):
#     y=predicted_labels[i]
#     x=train_labels[i]
#     confusion_matrix[int(x)][int(y)]+=1
# print(confusion_matrix)



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

# mlp_classifier=MLPClassifier( 1465 0.998
#     hidden_layer_sizes=([40,40]),
#     activation='logistic',
#     solver='sgd',
#     alpha=0.001, #0.001 1463
#     batch_size=200, #200: 1439, .984
#     learning_rate='adaptive',
#     learning_rate_init=0.5,
#     power_t=0.5,#pt learning_rate=invscaling
#     max_iter=850,
#     shuffle=False,
#     random_state=None,
#     tol=0.0001,#+1 0
#     momentum=0.7,
#     early_stopping=False,
#     validation_fraction=0.3,
#     n_iter_no_change=10
# )

# mlp_classifier=MLPClassifier( #1454 0.998
#     hidden_layer_sizes=(20,20),
#     activation='relu',
#     solver='sgd',
#     alpha=0.00001,
#     batch_size=200,
#     learning_rate='adaptive',
#     learning_rate_init=0.003,
#     power_t=0.5,#pt learning_rate=invscaling
#     max_iter=950,
#     shuffle=True,
#     random_state=None,
#     tol=0.00003,
#     momentum=0.7,
#     early_stopping=False,
#     validation_fraction=0.1,
#     n_iter_no_change=10
# )