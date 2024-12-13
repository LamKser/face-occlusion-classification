[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)][1]
![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.9-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Wandb](https://img.shields.io/badge/Weights_&_Biases-000000?style=for-the-badge&logo=WeightsAndBiases&logoColor=yellow)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)

[1]: https://www.kaggle.com/datasets/dinhhoanglam/face-occlusion

# Face occlusion classification

```
git clone https://github.com/LamKser/face-occlusion-classification.git
cd face-occlusion-classification
```
<!-- * Tensorflow implement [here]() -->
## :computer: **Hardware & Environment** 
* All the train and test processes are done on google colab with GPU Tesla T4
    ```
    conda env create -f environment.yml
    conda activate face-occlusion
    ```

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
* :link: Data link: [face occlusion dataset](https://www.kaggle.com/datasets/dinhhoanglam/face-occlusion)

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
    python test_model.py --opt configs/test.yml
    ```
* Test single image

    ```
    python test_single_image.py --model <model_name> --weight <weight_path> --image <image_path>
    ```
* ONNX model
    * Convert pytorch model to onnx

        ```
        python onnx/convert_2_onnx.py --model <model name> \\
                                      --weight <weight and checkpoint file> \\
                                      --save <path/to/save/onnx/*.onnx> \\
                                      --opset_version <version> (optional)
        ```
    * Run onnx model

        ```
        python onnx/run_onnx.py --onnx <onnx file> --img <your image>
        ```
## :bar_chart: **Results** (Train/Val/Test)
* All the trained model: [trained model](https://drive.google.com/drive/folders/1-Dy6xcKH9D5YBeYCav_PZyJwYeZPSnuq?usp=sharing)
* The pretrained models are trained with **30 epochs**

## Last model
<table border>
    <tr align="center">
        <th align="left">Model</th>
        <!-- <th>Pretrained?</th> -->
        <th>Params (M)</th>
        <th>Infer (ms)</th>
        <th>Accuracy</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1</th>
        <th>Weights</th>
    </tr>
    <tr align="center">
        <td align="left">VGG16</td>
        <td>134.2</td>
        <td>7.76</td>
        <td>0.9805</td>
        <td>0.981</td>
        <td>0.9789</td>
        <td>0.9799</td>
        <td><a href="https://drive.google.com/file/d/11g4Qdyt-dsjc-9ahDWKpLII3CKzGeKiE/view?usp=sharing">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">VGG19</td>
        <td>139.5</td>
        <td>9.36</td>
        <td>0.9836</td>
        <td>0.9831</td>
        <td>0.9831</td>
        <td>0.9831</td>
        <td><a href="https://drive.google.com/file/d/1-QM-quWh8AVyPsbPKXuNFZuBbqsUop6W/view?usp=sharing">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">VGG16-BN</td>
        <td>134.2</td>
        <td>8.3</td>
        <td>0.9734</td>
        <td>0.9746</td>
        <td>0.9705</td>
        <td>0.9725</td>
        <td><a href="https://drive.google.com/file/d/11P9hmqoUgCzCYtrrmoJTZC30PxxUF6ln/view?usp=sharing">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">VGG19-BN</td>
        <td>139.5</td>
        <td>10.01</td>
        <td>0.9713</td>
        <td>0.9765</td>
        <td>0.9642</td>
        <td>0.9703</td>
        <td><a href="https://drive.google.com/file/d/11cAzmekS32dSOGIMiogq2m0cc7BQ903u/view?usp=drive_link">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">DenseNet169</td>
        <td>12.4</td>
        <td>25.46</td>
        <td>0.9795</td>
        <td>0.9729</td>
        <td>0.9852</td>
        <td>0.979</td>
        <td><a href="https://drive.google.com/file/d/1ylcXT7KLCauX_eeITw2yv823mR5uXq2u/view?usp=sharing">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">DenseNet201</td>
        <td>18</td>
        <td>31.06</td>
        <td>0.9744</td>
        <td>0.9787</td>
        <td>0.9684</td>
        <td>0.9735</td>
        <td><a href="https://drive.google.com/file/d/11p-_fr8a1EudS3g-IgGIsjPncUe1zMk4/view?usp=drive_link">link</a></td>
    </tr>
    <tr align="center">
        <td align="left">ResNet18</td>
        <td>11.1</td>
        <td>3.69</td>
        <td>0.9703</td>
        <td>0.9665</td>
        <td>0.9726</td>
        <td>0.9695</td>
        <td><a href="https://drive.google.com/file/d/10FmUWT0pAU3_Mwl7-eBadKM4xy75AEDi/view?usp=drive_link">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">ResNet50</td>
        <td>23.5</td>
        <td>7.15</td>
        <td>0.9754</td>
        <td>0.9787</td>
        <td>0.9705</td>
        <td>0.9746</td>
        <td><a href="https://drive.google.com/file/d/11n6KxAxIRdCgoKlIJsUjLsiHC-aOs1i6/view?usp=drive_link">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">ResNet152</td>
        <td>58.1</td>
        <td>19.31</td>
        <td>0.9805</td>
        <td>0.983</td>
        <td>0.9768</td>
        <td>0.9799</td>
        <td><a href="https://drive.google.com/file/d/10U27HvjHvbMPHZIuvT4rxuxuNDu1Mdkd/view?usp=drive_link">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">ConvNeXt-Base</td>
        <td>87.5</td>
        <td>13.26</td>
        <td>0.9867</td>
        <td>0.9894</td>
        <td>0.9831</td>
        <td>0.9862</td>
        <td><a href="https://drive.google.com/file/d/1R_HF1Psv3qji10uMbPyd8OnTR9SGor3J/view?usp=drive_link">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">ConvNeXt-Small</td>
        <td>49.4</td>
        <td>11.54</td>
        <td>0.9887</td>
        <td>0.9853</td>
        <td>0.9915</td>
        <td>0.9884</td>
        <td><a href="https://drive.google.com/file/d/1-BONvtmHbzxgoJvgxQ1Nt442iIWvZU0Z/view?usp=drive_link">link</a></td>
    </tr>
    <tr align="center">
        <td  align="left">ConvNeXt-Tiny</td>
        <td>27.8</td>
        <td>7.24</td>
        <td>0.9867</td>
        <td>0.9832</td>
        <td>0.9894</td>
        <td>0.9863</td>
        <td><a href="https://drive.google.com/file/d/10rCsarUFijUzz3roSN1bi5-mfz_37c2o/view?usp=sharing">link</a></td>
    </tr>
</table>
