[![Google Drive](https://img.shields.io/badge/Google%20Drive-00897B?style=for-the-badge&logo=google-drive&logoColor=white)][1]
![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.9-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Wandb](https://img.shields.io/badge/Weights_&_Biases-000000?style=for-the-badge&logo=WeightsAndBiases&logoColor=yellow)

[1]: https://drive.google.com/file/d/1Mq_OZJLHiedpc9D7O40aNIbsvJM8rx39/view?usp=share_link

# Face occlusion classification


```
git clone https://github.com/LamKser/face-occlusion-classification.git
cd face-occlusion-classification
```

## :computer: **Hardware**
* All the train and test processes are done on google colab with GPU Tesla T4

## **:books: Dataset** 

* Crawl 9,749 images from the internet, crop the face by using [FaceMaskDetection](https://github.com/AIZOOTech/FaceMaskDetection) and divide into 2 classes:
    * `0` - Non-occluded face
    * `1` - Occluded face
<div align="center">
    <img src="images/non_occluded_face.png" width=70%, height=70%>
    <div>
        <strong>Figure 1: </strong>Non-occluded face example
    </div>
</div>

<div align="center">
    <img src="images/occluded_face.png" width=70%, height=70%>
    <div>
        <strong>Figure 2: </strong>Occluded face example
    </div>
</div>

* Then split the dataset into 3 sets (7 - 2 - 1):
    * `Train set` : 6,826 images
    * `Val set` : 1,945 images
    * `Test set` : 978 images

* Data structure:

    ```
    face_occlusion
    ├───Train
    │   ├───1
    │   │   ├─face_0.jpg
    │   │   ├─face_1.jpg
    │   │   └...
    │   └───0
    ├───Val
    │   ├───1
    │   └───0
    └───Test
        ├───1
        └───0
    ```
* :link: Data link: [face occlusion dataset](https://drive.google.com/file/d/1Mq_OZJLHiedpc9D7O40aNIbsvJM8rx39/view?usp=share_link)

## :triangular_ruler: **Config**
* To use other model or change hyperparameters, you can edit `train.yml` and `test.yml` in `configs` folder
* Available models: `densenet169`, `resnet18`, `resnet50`


## **:building_construction: Train model**
* Train
    ```
    python train.py --opt configs/train.yml
    ```

* Show the training and validation progress
    ```
    tensorboard --logdir logger
    ```

* If using `wandb` to log training process:
    ```
    wandb:
        project: <Type your project>
        name: <Type experiment name>
    ```
    
## **:chart_with_upwards_trend: Test model**
* Test the model

    ```
    python test.py --opt configs/test.yml
    ```
* Test single image

    ```
    python test_single_image.py --model <model_name> --weight <weight_path> --image <image_path>
    ```
## :bar_chart: **Results** (Train/Val/Test)
* All the trained model: [trained model]()
* Results: [Results.md](Results.md)

                          


<!-- https://pytorch.org/vision/stable/models.html -->
<!-- https://www.w3schools.com/icons/tryit.asp?filename=tryicons_fa-close -->
<!-- https://www.w3schools.com/icons/tryit.asp?filename=tryicons_fa-check -->