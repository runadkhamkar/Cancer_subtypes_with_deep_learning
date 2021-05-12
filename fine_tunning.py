from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization,Concatenate
from tensorflow.keras.utils import plot_model
from keras.models import load_model,Model,Sequential
from sklearn.preprocessing import  label_binarize
from matplotlib import pyplot
import pickle
def fine_tune(x_train,y_train,x_test,y_test,model_given,filename,epochs=50):
	y_train_binarize = label_binarize(y_train, classes=[0,1,2,3])
	y_test_binarize = label_binarize(y_test, classes=[0,1,2,3])
	model=load_model(model_given)
	model.summary()
	def print_layer_trainable(model):
		for layer in model.layers:
			print("{0}:\t{1}".format(layer.trainable,layer.name))
	#print_layer_trainable(model)
	for l in model.layers[1:-1]:
		#print(l.name)
		l.trainable=False
	#print_layer_trainable(model)
	x=model.output
	new=Dense(4,activation="softmax",name="output")(x)
	final = Model(inputs=model.input, outputs=new)
	#plot_model(final, 'fine_tunning.png', show_shapes=True)
	#print_layer_trainable(final)
	final.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy')
	final.fit(x_train, y_train_binarize, batch_size = 20, epochs = epochs)
	#score = final.evaluate(x_test, y_test_binarize, verbose=1, batch_size=20)
	#print("Test Accuracy:",score)
	#model2 = Model(model.input,model.layers[:-1])
	x = Lambda(lambda x:x)(model.layers[-1].output)
	model2 = Model(inputs=model.input, outputs=[x])
	#model.summary()
	model2.summary()
	for i in model2.layers:
		i.trainable=True
	model2.save(filename)