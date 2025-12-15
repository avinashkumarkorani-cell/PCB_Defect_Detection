import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io
import torch
import hashlib
import time


# --- Setup and Configurations ---
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🤖",
    layout="wide",
)


# --- Defect classes and their solutions ---
DEFECT_SOLUTIONS = {
    "Missing_hole": {
        "description": "A hole that was supposed to be drilled is absent.",
        "solution": "1. Review the drilling program for errors. 2. Manually drill the missing hole if possible. 3. Re-fabricate the board if the defect is critical."
    },
    "Mouse_bite": {
        "description": "A small indentation or nick on the edge of a trace or pad.",
        "solution": "1. Check for mechanical damage during handling. 2. For minor bites, the board may pass quality checks. 3. For severe bites, the board must be rejected to prevent an open circuit."
    },
    "Open_circuit": {
        "description": "A break in a copper trace, preventing electrical flow.",
        "solution": "1. Use a continuity tester to confirm the break. 2. Repair by soldering a jumper wire across the break. 3. If repair is not feasible, the board must be discarded."
    },
    "Short": {
        "description": "Two separate traces are accidentally connected, creating a path for electrical flow where there should not be one.",
        "solution": "1. Use a multimeter to locate the short. 2. Carefully scrape the copper to remove the short. 3. Check for any solder bridges or foreign material causing the short."
    },
    "Spur": {
        "description": "A sharp protrusion extending from a trace, potentially causing a short with other components.",
        "solution": "1. Carefully scrape the copper to remove the spur. 2. Use a microscope to ensure the entire spur has been removed. 3. Check for shorts after repair."
    },
    "Spurious_copper": {
        "description": "Unwanted copper residue that may cause shorts or other issues.",
        "solution": "1. Carefully remove the spurious copper using appropriate etching or mechanical methods. 2. Check surrounding areas for additional copper residue. 3. Test for shorts after removal."
    }
}


# --- Model Loading ---
@st.cache_resource
def load_model():
    """Load the trained YOLOv8 model from the .pt file."""
    model_path = 'pcb_defect_detection_model.pt'
    try:
        model = YOLO(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: The model file '{model_path}' was not found. Please ensure it is in the same directory as this app.")
        return None


# --- Main Page Functions ---
def show_home_page():
    st.title("PCB Defect Detection 🔍")
    st.markdown("### Welcome to the PCB Defect Detector")
    st.write("This application uses a trained YOLOv8 model to automatically identify common defects on Printed Circuit Boards (PCBs).")
    
    st.markdown("---")
    st.header("What Our Model Detects")
    st.write("Our model is trained to recognize the following types of PCB defects:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Defect Types**")
        for defect in DEFECT_SOLUTIONS.keys():
            st.write(f"- **{defect.replace('_', ' ')}**")
    
    with col2:
        st.markdown("**Description and Solutions**")
        for defect, details in DEFECT_SOLUTIONS.items():
            with st.expander(f"**{defect.replace('_', ' ')}**"):
                st.markdown(
                    f"**Description:** {details['description']}\n\n"
                    f"**Suggested Solution:** {details['solution']}"
                )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Account")
    if st.sidebar.button("Sign Up"):
        st.session_state.page = "signup"
        st.rerun()
    if st.sidebar.button("Log In"):
        st.session_state.page = "login"
        st.rerun()


def show_signup_page():
    st.title("Sign Up")
    st.write("Create a new account to access the PCB Defect Detector.")
    
    with st.form("signup_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        signup_submitted = st.form_submit_button("Sign Up")
        
        if signup_submitted:
            if not new_username or not new_password:
                st.error("Username and password cannot be empty.")
            elif new_username in st.session_state.users:
                st.error("Username already exists. Please choose a different one.")
            else:
                st.session_state.users[new_username] = hashlib.sha256(new_password.encode()).hexdigest()
                st.success("Account created successfully! You can now log in.")
                st.session_state.page = "login"
                st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update(page="home"))
    st.sidebar.button("Already have an account? Log In", on_click=lambda: st.session_state.update(page="login"))


def show_login_page():
    st.title("Log In")
    st.write("Log in with your credentials to access the PCB Defect Detector.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")
        
        if submitted:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if username in st.session_state.users and st.session_state.users[username] == password_hash:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "prediction"
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    st.sidebar.markdown("---")
    st.sidebar.button("Back to Home", on_click=lambda: st.session_state.update(page="home"))
    st.sidebar.button("Need an account? Sign Up", on_click=lambda: st.session_state.update(page="signup"))


def show_prediction_page():
    model = load_model()
    if model is None:
        return

    st.title('PCB Defect Detection with YOLOv8 🤖')
    st.write('Upload an image of a Printed Circuit Board to detect any defects.')
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button('Detect Defects'):
                with st.spinner('Running detection...'):
                    results = model(image, imgsz=640, conf=0.25)
                    
                    if results:
                        r = results[0]
                        im_array = r.plot()
                        annotated_image = Image.fromarray(im_array[..., ::-1])
                        
                        st.header('Detection Results')
                        st.image(annotated_image, caption='Image with Detected Defects', use_column_width=True)
                        
                        if r.boxes:
                            st.subheader("Defects Detected:")
                            detected_defects = []
                            for box in r.boxes:
                                class_name = model.names[int(box.cls[0])]
                                confidence = float(box.conf[0])
                                st.write(f"- **{class_name.replace('_', ' ')}**: Confidence: {confidence:.2f}")
                                detected_defects.append(class_name)
                            
                            # Debug information
                            st.write("**Debug Info:**")
                            st.write("Detected defects:", detected_defects)
                            st.write("Available solutions:", list(DEFECT_SOLUTIONS.keys()))

                            st.markdown("---")
                            st.subheader("Suggested Solutions:")
                            
                            # Check each unique detected defect
                            unique_defects = set(detected_defects)
                            solutions_found = False
                            
                            for defect in unique_defects:
                                st.write(f"Looking for solution for: '{defect}'")
                                
                                # Try exact match first
                                if defect in DEFECT_SOLUTIONS:
                                    solutions_found = True
                                    with st.expander(f"**Solution for {defect.replace('_', ' ')}**"):
                                        st.write("**Suggested Repair Steps:**")
                                        st.write(DEFECT_SOLUTIONS[defect]['solution'])
                                        st.warning("⚠️ **Important Note:** These are general suggestions only. Please take proper precautions, use appropriate safety equipment, and conduct your own research or consult with qualified professionals before attempting any repairs. Always follow industry standards and safety protocols when working with electronic components.")
                                
                                # Try case-insensitive match
                                else:
                                    for solution_key in DEFECT_SOLUTIONS.keys():
                                        if defect.lower() == solution_key.lower():
                                            solutions_found = True
                                            with st.expander(f"**Solution for {defect.replace('_', ' ')}**"):
                                                st.write("**Suggested Repair Steps:**")
                                                st.write(DEFECT_SOLUTIONS[solution_key]['solution'])
                                                st.warning("⚠️ **Important Note:** These are general suggestions only. Please take proper precautions, use appropriate safety equipment, and conduct your own research or consult with qualified professionals before attempting any repairs. Always follow industry standards and safety protocols when working with electronic components.")
                                            break
                                    else:
                                        st.warning(f"No solution available for defect type: '{defect}'")
                            
                            if not solutions_found:
                                st.info("No matching solutions found for the detected defects.")
                                
                        else:
                            st.info("No defects were detected in this image.")
                    else:
                        st.warning("Could not get a result from the model.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")


# --- Main App Logic ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "home"
if "users" not in st.session_state:
    # Initialize with hard-coded users
    st.session_state.users = {
        "testuser": hashlib.sha256("password123".encode()).hexdigest(),
        "john.doe": hashlib.sha256("securepass".encode()).hexdigest()
    }
    
st.sidebar.title("Navigation")
if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Go to Detector"):
        st.session_state.page = "prediction"
        st.rerun()
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "home"
        st.rerun()
else:
    st.sidebar.info("You must log in to use the detector.")

# Display the correct page based on session state
if st.session_state.page == "home":
    show_home_page()
elif st.session_state.page == "signup":
    show_signup_page()
elif st.session_state.page == "login":
    show_login_page()
elif st.session_state.logged_in and st.session_state.page == "prediction":
    show_prediction_page()
else:
    # Default to home if state is inconsistent
    st.session_state.page = "home"
    st.rerun()
