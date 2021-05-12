
# train autoencoder for classification with no compression in the bottleneck layer
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
def getFeatures(path,X_train,y_train,X_test,y_test,set_epochs=50):
	# define encoder
	n_inputs=X_train.shape[1]
	visible = Input(shape=(n_inputs,))
	# encoder level 1
	e = Dense(1000)(visible)
	# encoder level 2
	e = Dense(150)(e)
	e = Dense(50)(e)
	n_bottleneck = 50
	bottleneck = Dense(n_bottleneck,name="bottleneck")(e)
	# define decoder, level 1

	d = Dense(50)(bottleneck)
	d = Dense(150)(d)
	d = Dense(1000)(d)
	# output layer
	output = Dense(n_inputs, activation='linear')(d)
	# define autoencoder model
	model = Model(inputs=visible, outputs=output)
	# compile autoencoder model
	model.compile(optimizer='adam', loss='mse')
	# plot the autoencoder
	plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
	# fit the autoencoder model to reconstruct input
	history = model.fit(X_train, X_train, epochs=set_epochs, batch_size=16, verbose=2, validation_data=(X_test,X_test))
	# plot loss
	print(bottleneck,n_bottleneck)
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()
	# define an encoder model (without the decoder)
	encoder = Model(inputs=visible, outputs=bottleneck)
	plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
	# save the encoder to file
	encoder.summary()
	return encoder