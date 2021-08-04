# Cancer_subtypes_with_deep_learning
Documentation are configured in HTML format with direct links to source files and method defination in each file
	To read the documentation in linux:
		run file ./Documentation/show.sh
	To read the documentation in Windows:
		open file at path in your browser "./Documentation/_build/html/index.html"
		
Below are the libraries required for the proposed project.
• Keras 2.0.6
• Tensorflow(1.13.1)
• Scikit-Learn(0.20.3)
• Numpy(1.16.3)
• Imbalanced-Learn(0.4.3)

Directory structure is given in ./Sorce_code/tree.txt file for better understanding.

Project contains 5 deep learning models:
	1.Principle component Analysis
		to run use following command
		$python PCA_Classification.py
	2.SelectBest feature extractor
		to run use following command
		$python SelectBest_Classification.py
	3.Variational AutoEncoder
		to run use following command
		$python Variational_multilayer.py
	4.With Fine Tunning extractor
		to run use following command
		$python WithFineTunning.py
	5.Without Fine Tunning extractor
		to run use following command
		$python WithoutFineTunning.py

Datasets are stored in data folder
	To change the dataset uncomment required dataset and comment other or can add new dataset and give path to executaing files. 

Trained models are stored in defined folders automatically if you gave permissions to directory else you have create folders by yourself and add path to executing file

You can add or remove classifiers using getClassifier files (see the documentation).
