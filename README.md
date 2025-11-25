# RePo
# We will supplement the complete code and its running scripts as soon as possible.
This code implements the main algorithms and experiments from our AAAI 2026 paper "Region-Point Joint Representation for Effective Trajectory Similarity Learning".
## Require Packages
All required packages and their versions are listed in requirements.txt.

Install them using the following command:
`pip install -r requirements.txt
`

## Data

These publicly available datasets can be easily found and downloaded online by searching their official names:

- **Porto Dataset**
- **GeoLife Dataset**
- **Chengdu Dataset**

The data folder contains data that has already been cleaned and augmented, with the original latitude and longitude information stored in the 7th and 8th dimensions, respectively.



For all datasets, we retain trajectories with lengths between 10 and 300 points. Trajectory distance matrix is computed by traj_dist. 


To download remote images automatically, run the following script:

```bash
python utils/download_remoteimages.py
```
Note: Please adjust the parameters (e.g., latitude/longitude range, zoom level, save directory) inside the download_remoteimages.py script according to your needs before running.
## Preprocessing
Run `utils/preprocess.py`. This script processes raw trajectory data and map tiles to generate enriched trajectory features and embeddings needed for training.
Adjust the config file for paths and parameters as needed. The output files will be saved according to the config settings.

## Training & Evaluating
```bash
python train.py --dataset porto --metric dtw
```
This command trains the RePo model on the Porto dataset using the Dynamic Time Warping (DTW) metric for trajectory similarity evaluation.
Training and evaluation settings (e.g., epochs, learning rate) can be adjusted in the corresponding config files.

After training, the model checkpoints(in exp/wts) and evaluation results will be saved automatically.
