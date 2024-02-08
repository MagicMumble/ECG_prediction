# Detecting of the Acute Myocardial Infarction based on the electrocardiograms using Convolutional Neural Network
(Master's thesis)

Acute myocardial infarction (AMI), commonly known as heart attack, requires timely detection to prevent severe health-related complications. This study aims to foster the development of automated AMI recognition systems based solely on electrocardiogram (ECG).

To this aim, 12-lead ECG recordings were obtained from the UK Biobank and gender-specific datasets were prepared and labelled according to cardiologists' diagnoses. Two groups were defined: AMI patients and healthy individuals. Addressing class imbalance, undersampling was performed, and features were extracted using Continuous Wavelet Transform (CWT) to generate scalograms. Convolutional neural networks (CNNs) were trained on raw ECGs and scalograms, with better performance observed for CWT-preprocessed data. For males, CNNs trained on scalograms achieved 70% sensitivity, 88% specificity, 66% F1-score, 84% accuracy, and an AUC score of 0.83. For females, the model yielded 57% sensitivity, 78% specificity, 17% F1-score, 77% accuracy, and an AUC score of 0.73.

This research shows the potential of automated AMI prediction tools based on ECGs. Embedding such algorithms in sensors could enable cost-effective and non-invasive monitoring, benefiting high-risk groups like those with familial heart conditions. Additionally, these algorithms could assist doctors by identifying readouts that warrant extra investigation.

The `readingTheDiagnoses.ipynb` jupyter notebook contains the methods analysing the data, preprocessing and tranforming it. Also, the CNN models are build and trained in this notebook. The plots are build for visualization.

To run the jupyter notebook, the Python packages listed in the `conda_list` file should be installed. The Python of version 3.11.5 was used to run the notebook.
