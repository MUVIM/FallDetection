# MUVIM
 
This code is meant to allow you to train a model on the MUVIM dataset. The dataset must be downloaded seperately.

In order to efficently load data quicker it must be made in to an H5PY file format. This can be done with the provided datasetcreator script. In order for this to work you must have your files organized in the following format. A main directory for the camera that contains two folders - fall and nonfall. Each one of these folders contains a folder for each video - Fall1, Fall2... etc. (The same for NonFall - NonFall1, NonFall2..etc). These folders then contain a series of images that correspond to video frames. 

Additionally paths for the location of the H5PY files must be updated within the script. 
Locations for the datastorage must also be updated including the location of CSVs containg labels. The CSV file should be located in the same folder as the Falls and NonFalls. Please reference the MUVIM dataset labels CSVs to see how they should be formated. You can specify parameters here such as inpainting and image size. In order to run correctly, additional project directorys and files must also be included as shown in the directory.


Once you have a dataset created you can run one of the models in order to train and test.


