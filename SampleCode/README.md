This code is meant to allow you to train a model on the MUVIM dataset. The dataset must be downloaded separately.

PLEASE DO NOT EMAIL FOR INSTRUCTIONS ON CODE TROUBLE SHOOTING.

Code is provided as a starting point and to help allow other to train models easier.

Once data is downloaed, it must be made in to an H5PY file format for efficient loading. This can be done with the provided dataset creator script. This scripts expects files to be organized in the following format. A main directory for the camera that contains two folders - Fall and NonFall. Each one of these folders contains a folder for each video - Fall1, Fall2... etc. (The same for NonFall - NonFall1, NonFall2..etc). These folders then contain a series of images that correspond to video frames.

Example:

CameraName(Main Directory)
- labels.csv
- Fall
  - Fall1
    - Image1
    - Image2
    - …
  - Fall2
  - ….
- NonFall
  - NonFall1
  - NonFall2
  - …

Datapaths for the location of the H5PY files must be updated within the script. Locations for the data storage must also be updated including the location of CSVs containing labels. The CSV file should be in the same folder as the Falls and NonFalls. Please reference the MUVIM dataset labels CSVs to see how they should be formatted. You can specify parameters here such as inpainting and image size. In order to run correctly, additional project directory's and files must also be included as shown in the directory.

The notebook file:

single\_modality.ipynb

Can be used to train and test models. Example output from this script is shown. It imports many functions from the functions.py file.
