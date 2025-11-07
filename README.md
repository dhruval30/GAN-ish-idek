# GenDet - RealityGAN: A First Working Prototype

so we built this thing to make fake human faces. the dataset we're using contains 5000 real images of human faces, which is the main data we train on. it also has 4630 ai-generated images, which aren't that useful for now, but we'll let them be. the project aims to train a gan that can create synthetic data that's indistinguishable from the real thing, even to a forensic detector.

### 1\. Project Structure

the project is organized in a standard, professional layout. all of the key files are located in the following directories:

  * `src/train.py`: the main training loop and entry point for the project.
  * `src/models/`: contains the architecture definitions for the `generator.py` and `discriminator.py` networks.
  * `src/utils/`: includes the `data_loader.py` script for efficient data handling.
  * `data/`: the root directory for all datasets.
  * `output/`: the directory where all generated images and model checkpoints are saved.
  * `progression.mp4`: a video showing the model's learning progress over 50 epochs.

### 2\. Local Setup & Requirements

#### Prerequisites

  * **Python:** Python 3.8+
  * **Libraries:** `torch`, `torchvision`, `numpy`
  * **Hardware:** An Apple Silicon Mac (using the `mps` device) or an NVIDIA GPU with CUDA (will need to configure somethings for that to work).

#### Installation

1.  clone this repository and navigate into the project directory.

2.  set up a virtual environment and install the required libraries:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install torch torchvision numpy
    ```

#### Dataset

1.  download the dataset from this link and place the folder containing the images in the root directory of your project.
2.  extract the images. You should have a folder containing all the `JPEG` images.
3. you will have two directories AI-Generated Images and Real Images
4. now run the `notebooks/datasetReview.ipynb` which will resize the images to 128*128px and saves that to `data/processed/resized/faces/` which the script expects.
5. now you have successfully loaded the data and can begin.

#### Model Checkpoints

the final trained model is too large for GitHub. the download link is provided in a text file at the root of the project. 

1.  look for a text file in the project's root directory. `modelWeights.txt`
2.  follow the link to download the `gan_checkpoint_epoch_50.pth` file.
3.  place this file in the `output/checkpoints/` directory.

### 3\. How to Run the Project

#### To Continue Training

you can continue training the model from where it left off. open `src/train.py` and change the `if __name__ == '__main__':` block to load the last checkpoint and set the `start_epoch` accordingly.

```bash
python src/train.py
```

#### To Run Inference (Generate New Images)

the `inference.py` script is set up to load a saved checkpoint and generate new images.

1.  open `src/inference.py`.

2.  edit the `CHECKPOINT_PATH` to point to the desired `.pth` file (e.g., `gan_checkpoint_epoch_50.pth`).

3.  change `NUM_IMAGES` to `1` for a single image, or `64` for a grid.

    ```bash
    python src/inference.py
    ```

### 4\. About This Prototype

this prototype successfully implements a stable `WGAN-GP` training process with a custom dual-detector discriminator, with a forensic detector. it demonstrates a working solution to the problem of training instability and is a strong foundation for future research into producing truly forensically indistinguishable synthetic data.

![Model Training Progression](progression.gif)
