# hair_unet
This is an application of the u-net convolutional neural network architecture to segmenting hairstyles from a set of images from the 1900s to the present. Hair segmentation is a much studied application of neural networks (see [Arabi 2015](https://patents.google.com/patent/US9928601B2/en), [Muhammed et al. 2018](https://www.researchgate.net/profile/Umar_Riaz_Muhammad/publication/323212461_Hair_detection_segmentation_and_hairstyle_classification_in_the_wild/links/5d9c94b2299bf1c36301d9a1/Hair-detection-segmentation-and-hairstyle-classification-in-the-wild.pdf), and [Chai et al. 2016](http://eprints.whiterose.ac.uk/134268/1/TianjiaShao_AutoHair.pdf) for some recent examples). To my knowledge, the dataset and model provided here are unique in that they incorporate historical data to represent photography and hairstyles from the past century in the United States.

## Requirements
This code was last tested with Tensorflow 1.14.0 and Keras 2.3.0.

## Credits
U-net model code is adapted from [zhixuhao's repo](https://github.com/zhixuhao/unet). Please contact me at andronovohpf@gmail.com if you are interested in obtaining the trained model file.

## Data sources
The dataset contains image-mask pairs from three sources:
+ [The Figaro1K dataset](http://projects.i-ctm.eu/it/progetto/figaro-1k) containing 1050 naturalistic images of hair. In the dataset, these images are named with the prefix "Frame_". These images are center-cropped, downsampled to 256x256, and converted to grayscale for parity with the other data sources.
+ A set of hand-segemented yearbook photos from a custom dataset I created scraping and cropping American high school yearbook images from [archive.org](archive.org). These hand-labeled images are named with the prefix "19x0_crop" where x is an integer such that 0 < x < 10.
+ A set of images from the same custom dataset, curated from several rounds of dataset augmentation by manually selecting high-quality segmentation instances on a test dataset and adding these image-mask pairs into the training set for another round of training. Each round is indexed alphabetically as file prefixes ("a_" marks the first round of augmentation, "b_" the second, etc.).

## Running the model
`python main.py` is all it takes.
