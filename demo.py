from io import BytesIO
import yaml

# from PIL import Image
import streamlit as st
# import torch
# from torchvision import transforms

# from model import Model
# from utils import load_weight
from src.test import TestModel

# Configs
with open("configs/demo.yml") as f:
    config_demo = yaml.load(f, Loader=yaml.SafeLoader)

with open("configs/test.yml") as f:
    config_model = yaml.load(f, Loader=yaml.SafeLoader)

CLASSES = {0: "non-occluded",
           1: "occluded"}

def run_model(configs, image):
    return TestModel(configs).run(image)


def main():
    st.set_page_config(
        page_title="Face occlusion demo",
        layout="wide"
    )
    st.title("Face occlusion demo")
    col1, col2, col3 = st.columns(3)

    with col1:
        select_model = col1.selectbox("Choose model", config_demo["model"]["name"].keys())
        config_model["model"]["name"] = select_model.lower()
        config_model["weight"] = config_demo["model"]["name"][select_model]


        up_img = col1.file_uploader("Upload face image")
        
        # device = "cuda" if torch.cuda.is_available() else "cpu"

    with col2:
        if up_img:
            st.image(up_img, width=300)
            pro, pre = run_model(config_model, up_img)

    with col3:
        if up_img:
            with st.spinner(text="Wait for it..."):
                st.write(
                    {
                        "model": select_model,
                        "prob": f"{round(pro.item() * 100, 2)} %",
                        "label": CLASSES[pre.item()]
                    }
                )

if __name__ == "__main__":
    main()

