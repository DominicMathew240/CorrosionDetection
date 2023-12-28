# Python In-built packages
from pathlib import Path
import PIL
from PIL import Image
import yaml 
from yaml.loader import SafeLoader
import base64
from io import BytesIO

# External packages
import streamlit as st
import streamlit_authenticator as stauth

# Local Modules
import settings
import helper
import analytics

# Setting page layout
st.set_page_config(
    page_title="Stratetics Experts",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="expanded"
)

# Image Logo Initialization
node_comp = Image.open("logo/node-comp.png")
stratetics = Image.open("logo/stratetics.png")
# dronez = Image.open("logo/dronez.png")

def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

node_comp_base64 = get_image_base64(node_comp)
stratetics_base64 = get_image_base64(stratetics)
# dronez_base64 = get_image_base64(dronez)

# Display images in the footer
footer = f"""
<style>
.footer {{
    background-color: #ffffff;
    position: fixed;
    right: 0;
    float: right;
    top: 0;
    width: 26%;
    border-radius: 0px 0px 0px 12px;
    text-align: center;
    z-index: 1000;
}}

.footer-image {{
    margin: 6px;
}}

</style>
<div class="footer">
<img src="data:image/png;base64,{stratetics_base64}" width="32%" class="footer-image"/>
<img src="data:image/png;base64,{node_comp_base64}" width="32%" class="footer-image"/>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            header {visibility: hidden;}
            .container {display:none;}
            [href*="https://streamlit.io/cloud"] {display: none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# load hashed passwords
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

names, authentication_status, username = authenticator.login("Login", "main")

if st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
elif st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar', key='unique_key')
    # Sidebar
    st.sidebar.header("Model Configuration")

    # Model Options
    model_type = st.sidebar.radio(
        "Select Task", ['Detection'])
        # "Select Task", ['Detection', 'Segmentation'])

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 0, 100, 30)) / 100

    # Selecting Detection Or Segmentation
    if model_type == 'Detection':
        model_path = Path(settings.DETECTION_MODEL)
    elif model_type == 'Segmentation':
        model_path = Path(settings.SEGMENTATION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)

    source_img = None
    global res
    res = None

    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                            use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                        use_column_width=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    #return res value so that can be used in global scope
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    # print(boxes)
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                        
        if res is not None:
            for r in res:
                number = r.boxes.cls
                confidence = r.boxes.conf
                class_name = r.boxes.cls

            # call the class distribution function from analytics.py
            analytics.display_class_distribution(number)

            # Error handling for empty predictions
            if len(number) <= 0:
                print()
            else:
                # Create two columns
                col3, col4 = st.columns(2)
    
                # Display the outputs in the columns
                with col3:
                    analytics.display_confidence_distribution(confidence)
                with col4:
                    analytics.display_confidence_heatmap(confidence, number)
    
                # Display the prediction summary
                analytics.display_prediction_summary(number, confidence)
            
    elif source_radio == settings.VIDEO:
        helper.play_stored_video(confidence, model)

    elif source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model)

    elif source_radio == settings.RTSP:
        helper.play_rtsp_stream(confidence, model)

    elif source_radio == settings.YOUTUBE:
        helper.play_youtube_video(confidence, model)

    else:
        st.error("Please select a valid source type!")
