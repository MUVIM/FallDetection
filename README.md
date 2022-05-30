# Version1
 
This code is meant to allow you to train a model on the MUVIM dataset. The dataset must be downloaded seperately.

In order to access the data quicker when training models it must be made in to an H5PY file format. This can be done with the datasetcreator script. In order for this to work you must have your files organized in the following format:

Camera
- Falls and Non Falls
 - numbered folders containing the video broken down into images
 (ex. Fall1, Fall2, Fall3 ... Fall35... etc) 

You must change paths and name of dataset for this script to work properly along with the correct file structure. This includes a labels CSVs. The CSV file should be located in the same folder as the Falls and NonFalls main directory. Please reference the MUVIM dataset labels CSVs to see how they should be formated. You can specify parameters here such as inpainting and image size.
Once you have a dataset created you can run one of the models in order to train and test. Results will be provided in a CSV.

In order to run correctly, additional project directorys and files must also be included.
