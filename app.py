import streamlit as st
import pandas as pd
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Wood Quality Prediction",
    page_icon="🪵",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🪵 Wood Quality Prediction System")

st.markdown("""
### AI-Based Wood Quality Analysis

Upload a wood image and enter the available wood parameters.

The system analyzes:

- Wood image
- Color
- Texture
- Brightness
- Edge density
- Possible crack indication
- Possible defect indication
- Physical wood parameters

The final result is:

**🟢 Good | 🟡 Medium | 🔴 Poor**
""")


# ============================================================
# DEMO DATASET
# ============================================================

data = [

    # --------------------------------------------------------
    # TEAK
    # --------------------------------------------------------

    [
        "Teak", 25, 11.5, 660, 78, 2.6, 25,
        "Light", "Straight", "Smooth", "Fine",
        "None", "None", 1, 90, 9,
        90, 20, 10, 8, 5, 5,
        "Good"
    ],

    [
        "Teak", 20, 13, 650, 75, 2.5, 25,
        "Medium", "Straight", "Smooth", "Fine",
        "None", "Low", 2, 85, 8,
        80, 25, 12, 10, 7, 6,
        "Good"
    ],

    [
        "Teak", 10, 19, 620, 70, 2.3, 25,
        "Medium", "Curved", "Moderate", "Medium",
        "Minor", "Medium", 3, 72, 7,
        65, 30, 16, 15, 10, 8,
        "Medium"
    ],

    [
        "Teak", 5, 29, 550, 55, 1.9, 25,
        "Dark", "Irregular", "Rough", "Coarse",
        "Major", "High", 6, 50, 5,
        45, 45, 25, 28, 18, 15,
        "Poor"
    ],


    # --------------------------------------------------------
    # PINE
    # --------------------------------------------------------

    [
        "Pine", 20, 11, 520, 62, 2.2, 25,
        "Light", "Straight", "Smooth", "Fine",
        "None", "None", 1, 88, 8,
        88, 22, 11, 9, 5, 5,
        "Good"
    ],

    [
        "Pine", 15, 14, 500, 58, 2.0, 25,
        "Light", "Straight", "Smooth", "Medium",
        "Minor", "Low", 2, 82, 8,
        78, 28, 14, 12, 8, 6,
        "Good"
    ],

    [
        "Pine", 10, 19, 480, 52, 1.8, 25,
        "Medium", "Curved", "Moderate", "Medium",
        "Minor", "Medium", 3, 70, 7,
        62, 34, 18, 16, 11, 9,
        "Medium"
    ],

    [
        "Pine", 7, 24, 450, 45, 1.7, 25,
        "Dark", "Irregular", "Rough", "Coarse",
        "Major", "High", 6, 52, 5,
        42, 48, 28, 30, 20, 17,
        "Poor"
    ],


    # --------------------------------------------------------
    # OAK
    # --------------------------------------------------------

    [
        "Oak", 30, 9.5, 720, 92, 3.1, 25,
        "Light", "Straight", "Smooth", "Fine",
        "None", "None", 1, 94, 10,
        94, 18, 8, 7, 4, 4,
        "Good"
    ],

    [
        "Oak", 25, 12.5, 710, 89, 3.0, 25,
        "Medium", "Straight", "Smooth", "Fine",
        "None", "Low", 2, 90, 9,
        88, 21, 10, 9, 5, 5,
        "Good"
    ],

    [
        "Oak", 18, 16, 680, 80, 2.7, 25,
        "Medium", "Straight", "Moderate", "Medium",
        "Minor", "Low", 2, 85, 8,
        76, 26, 13, 12, 7, 6,
        "Good"
    ],

    [
        "Oak", 12, 21, 640, 70, 2.5, 25,
        "Dark", "Curved", "Moderate", "Medium",
        "Minor", "Medium", 4, 70, 7,
        63, 32, 18, 17, 11, 9,
        "Medium"
    ],


    # --------------------------------------------------------
    # NEEM
    # --------------------------------------------------------

    [
        "Neem", 20, 13, 600, 72, 2.4, 25,
        "Light", "Straight", "Smooth", "Fine",
        "None", "None", 1, 87, 8,
        85, 24, 12, 10, 6, 5,
        "Good"
    ],

    [
        "Neem", 12, 17, 570, 65, 2.2, 25,
        "Medium", "Curved", "Moderate", "Medium",
        "Minor", "Medium", 3, 73, 7,
        68, 31, 16, 14, 9, 8,
        "Medium"
    ],

    [
        "Neem", 8, 22, 530, 55, 2.0, 25,
        "Dark", "Irregular", "Rough", "Coarse",
        "Minor", "High", 5, 58, 6,
        52, 40, 22, 25, 16, 13,
        "Medium"
    ],

    [
        "Neem", 5, 30, 480, 42, 1.7, 25,
        "Dark", "Irregular", "Rough", "Coarse",
        "Major", "High", 7, 48, 5,
        40, 52, 30, 32, 22, 18,
        "Poor"
    ],


    # --------------------------------------------------------
    # MAHOGANY
    # --------------------------------------------------------

    [
        "Mahogany", 30, 10, 650, 80, 2.8, 25,
        "Dark", "Straight", "Smooth", "Fine",
        "None", "None", 1, 94, 9,
        92, 18, 9, 7, 4, 4,
        "Good"
    ],

    [
        "Mahogany", 20, 13, 630, 75, 2.6, 25,
        "Medium", "Straight", "Smooth", "Medium",
        "Minor", "Low", 2, 86, 8,
        84, 23, 12, 10, 6, 5,
        "Good"
    ],

    [
        "Mahogany", 12, 19, 600, 68, 2.4, 25,
        "Medium", "Curved", "Moderate", "Medium",
        "Minor", "Medium", 3, 73, 7,
        70, 30, 16, 15, 9, 8,
        "Medium"
    ],

    [
        "Mahogany", 6, 28, 550, 55, 2.0, 25,
        "Dark", "Irregular", "Rough", "Coarse",
        "Major", "High", 6, 52, 5,
        45, 47, 27, 29, 19, 16,
        "Poor"
    ]
]


# ============================================================
# DATAFRAME
# ============================================================

columns = [

    "wood_type",
    "age",
    "moisture",
    "density",
    "hardness",
    "weight",
    "thickness",

    "color",
    "grain_pattern",
    "surface_condition",
    "texture",
    "defects",
    "crack_level",

    "knot_count",
    "previous_inspection_score",
    "durability_score",

    # Image features
    "image_brightness",
    "image_color_variation",
    "image_texture",
    "image_edge_density",
    "image_crack_score",
    "image_defect_score",

    "quality"
]


df = pd.DataFrame(
    data,
    columns=columns
)


# ============================================================
# FEATURE LISTS
# ============================================================

categorical_features = [

    "wood_type",
    "color",
    "grain_pattern",
    "surface_condition",
    "texture",
    "defects",
    "crack_level"
]


numeric_features = [

    "age",
    "moisture",
    "density",
    "hardness",
    "weight",
    "thickness",
    "knot_count",
    "previous_inspection_score",
    "durability_score",

    # Image features
    "image_brightness",
    "image_color_variation",
    "image_texture",
    "image_edge_density",
    "image_crack_score",
    "image_defect_score"
]


# ============================================================
# DATA TYPE CLEANING
# ============================================================

# Convert categorical columns to strings
for column in categorical_features:

    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
    )


# Convert numeric columns to numbers
for column in numeric_features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Replace invalid numerical values
for column in numeric_features:

    df[column] = df[column].fillna(
        df[column].median()
    )


# Target
df["quality"] = (
    df["quality"]
    .fillna("Unknown")
    .astype(str)
)


# ============================================================
# X AND Y
# ============================================================

X = df[
    categorical_features + numeric_features
]

y = df["quality"]


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore",
                dtype=np.float64
            ),

            categorical_features
        ),

        (
            "numeric",

            "passthrough",

            numeric_features
        )
    ]
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",

            RandomForestClassifier(

                n_estimators=200,

                random_state=42,

                class_weight="balanced"
            )
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST ACCURACY
# ============================================================

y_test_prediction = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_test_prediction
)


# ============================================================
# IMAGE FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_image_features(image_bytes):

    """
    Analyze uploaded wood image.

    Extracted features:

    1. Brightness
    2. Color variation
    3. Texture
    4. Edge density
    5. Crack score
    6. Defect score
    """

    # --------------------------------------------------------
    # Convert uploaded image to NumPy
    # --------------------------------------------------------

    image_array = np.asarray(
        bytearray(image_bytes),
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if image is None:

        return None


    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (500, 500)
    )


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # 1. BRIGHTNESS
    # --------------------------------------------------------

    brightness = float(
        np.mean(gray)
    )


    # --------------------------------------------------------
    # 2. COLOR VARIATION
    # --------------------------------------------------------

    color_variation = float(
        np.mean(
            np.std(
                image,
                axis=(0, 1)
            )
        )
    )


    # --------------------------------------------------------
    # 3. TEXTURE
    # --------------------------------------------------------

    texture = float(
        np.std(gray)
    )


    # --------------------------------------------------------
    # 4. EDGE DETECTION
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )


    edge_density = float(
        np.mean(edges > 0) * 100
    )


    # --------------------------------------------------------
    # 5. CRACK SCORE
    # --------------------------------------------------------

    dark_pixels = float(
        np.mean(gray < 70) * 100
    )


    crack_score = min(

        dark_pixels
        +
        edge_density * 0.3,

        100
    )


    # --------------------------------------------------------
    # 6. DEFECT SCORE
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    difference = cv2.absdiff(
        gray,
        blurred
    )


    defect_score = float(
        np.mean(difference)
    )


    defect_score = min(
        defect_score * 4,
        100
    )


    # --------------------------------------------------------
    # RETURN FEATURES
    # --------------------------------------------------------

    return {

        "image_brightness":
            round(brightness, 2),

        "image_color_variation":
            round(color_variation, 2),

        "image_texture":
            round(texture, 2),

        "image_edge_density":
            round(edge_density, 2),

        "image_crack_score":
            round(crack_score, 2),

        "image_defect_score":
            round(defect_score, 2)
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Model Information"
)

st.sidebar.write(
    "**Algorithm:** Random Forest Classifier"
)

st.sidebar.write(
    "**Trees:** 200"
)

st.sidebar.write(
    "**Classes:** Good / Medium / Poor"
)

st.sidebar.write(
    f"**Demo Test Accuracy:** "
    f"{accuracy * 100:.2f}%"
)

st.sidebar.warning(
    "The dataset is a small prototype dataset. "
    "The displayed accuracy should not be treated "
    "as real-world model accuracy."
)


# ============================================================
# STEP 1 — IMAGE UPLOAD
# ============================================================

st.header(
    "📷 Step 1 — Upload Wood Image"
)


uploaded_image = st.file_uploader(

    "Upload a wood image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# REQUIRE IMAGE
# ============================================================

if uploaded_image is None:

    st.info(
        "Please upload a wood image before prediction."
    )

    st.stop()


# ============================================================
# IMAGE ANALYSIS
# ============================================================

image_features = extract_image_features(

    uploaded_image.getvalue()
)


if image_features is None:

    st.error(
        "Unable to process the uploaded image."
    )

    st.stop()


# ============================================================
# DISPLAY IMAGE
# ============================================================

image_col1, image_col2 = st.columns(2)


with image_col1:

    st.subheader(
        "🖼️ Uploaded Wood Image"
    )

    st.image(

        uploaded_image,

        caption="Uploaded wood sample",

        use_container_width=True
    )


with image_col2:

    st.subheader(
        "🔍 Image Analysis"
    )

    st.write(
        "The image is analyzed using computer-vision "
        "features before prediction."
    )

    st.metric(
        "Brightness",
        image_features[
            "image_brightness"
        ]
    )

    st.metric(
        "Color Variation",
        image_features[
            "image_color_variation"
        ]
    )

    st.metric(
        "Texture",
        image_features[
            "image_texture"
        ]
    )

    st.metric(
        "Edge Density",
        f'{image_features["image_edge_density"]}%'
    )


# ============================================================
# STEP 2 — WOOD INFORMATION
# ============================================================

st.header(
    "📝 Step 2 — Wood Information"
)


col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    wood_type = st.selectbox(

        "Wood Type",

        [
            "Teak",
            "Pine",
            "Oak",
            "Neem",
            "Mahogany"
        ]
    )


    age = st.number_input(

        "Wood Age (years)",

        min_value=1,

        max_value=100,

        value=20
    )


    moisture = st.number_input(

        "Moisture Level (%)",

        min_value=0.0,

        max_value=100.0,

        value=12.0
    )


    density = st.number_input(

        "Density",

        min_value=100.0,

        max_value=1200.0,

        value=650.0
    )


    hardness = st.number_input(

        "Hardness",

        min_value=0.0,

        max_value=100.0,

        value=75.0
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    weight = st.number_input(

        "Weight (kg)",

        min_value=0.1,

        max_value=100.0,

        value=2.5
    )


    thickness = st.number_input(

        "Thickness (mm)",

        min_value=1.0,

        max_value=200.0,

        value=25.0
    )


    color = st.selectbox(

        "Color",

        [
            "Light",
            "Medium",
            "Dark"
        ]
    )


    grain_pattern = st.selectbox(

        "Grain Pattern",

        [
            "Straight",
            "Curved",
            "Irregular"
        ]
    )


    surface_condition = st.selectbox(

        "Surface Condition",

        [
            "Smooth",
            "Moderate",
            "Rough"
        ]
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    texture_input = st.selectbox(

        "Texture",

        [
            "Fine",
            "Medium",
            "Coarse"
        ]
    )


    defects = st.selectbox(

        "Defects",

        [
            "None",
            "Minor",
            "Major"
        ]
    )


    crack_level = st.selectbox(

        "Crack Level",

        [
            "None",
            "Low",
            "Medium",
            "High"
        ]
    )


    knot_count = st.number_input(

        "Knot Count",

        min_value=0,

        max_value=30,

        value=1
    )


    previous_score = st.number_input(

        "Previous Inspection Score",

        min_value=0.0,

        max_value=100.0,

        value=80.0
    )


    durability_score = st.number_input(

        "Durability Score",

        min_value=0.0,

        max_value=10.0,

        value=8.0
    )


# ============================================================
# STEP 3 — PREDICTION
# ============================================================

st.header(
    "🤖 Step 3 — AI Prediction"
)


if st.button(

    "🔎 Analyze Wood Quality",

    use_container_width=True
):

    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = pd.DataFrame({

        "wood_type": [
            wood_type
        ],

        "age": [
            age
        ],

        "moisture": [
            moisture
        ],

        "density": [
            density
        ],

        "hardness": [
            hardness
        ],

        "weight": [
            weight
        ],

        "thickness": [
            thickness
        ],

        "color": [
            color
        ],

        "grain_pattern": [
            grain_pattern
        ],

        "surface_condition": [
            surface_condition
        ],

        "texture": [
            texture_input
        ],

        "defects": [
            defects
        ],

        "crack_level": [
            crack_level
        ],

        "knot_count": [
            knot_count
        ],

        "previous_inspection_score": [
            previous_score
        ],

        "durability_score": [
            durability_score
        ],

        # ----------------------------------------------------
        # IMAGE FEATURES
        # ----------------------------------------------------

        "image_brightness": [

            image_features[
                "image_brightness"
            ]
        ],

        "image_color_variation": [

            image_features[
                "image_color_variation"
            ]
        ],

        "image_texture": [

            image_features[
                "image_texture"
            ]
        ],

        "image_edge_density": [

            image_features[
                "image_edge_density"
            ]
        ],

        "image_crack_score": [

            image_features[
                "image_crack_score"
            ]
        ],

        "image_defect_score": [

            image_features[
                "image_defect_score"
            ]
        ]
    })


    # ========================================================
    # CLEAN CATEGORICAL INPUT
    # ========================================================

    for column in categorical_features:

        input_data[column] = (

            input_data[column]

            .fillna("Unknown")

            .astype(str)
        )


    # ========================================================
    # CLEAN NUMERICAL INPUT
    # ========================================================

    for column in numeric_features:

        input_data[column] = pd.to_numeric(

            input_data[column],

            errors="coerce"
        )


    # ========================================================
    # REPLACE INVALID NUMBERS
    # ========================================================

    for column in numeric_features:

        if input_data[column].isna().any():

            input_data[column] = (

                input_data[column]

                .fillna(
                    df[column].median()
                )
            )


    # ========================================================
    # PREDICT
    # ========================================================

    prediction = model.predict(
        input_data
    )[0]


    probabilities = model.predict_proba(
        input_data
    )[0]


    classes = model.classes_


    probability_dict = dict(
        zip(
            classes,
            probabilities
        )
    )


    confidence = (
        max(probabilities) * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Final Prediction"
    )


    if prediction == "Good":

        st.success(
            "🟢 WOOD QUALITY: GOOD"
        )

    elif prediction == "Medium":

        st.warning(
            "🟡 WOOD QUALITY: MEDIUM"
        )

    else:

        st.error(
            "🔴 WOOD QUALITY: POOR"
        )


    st.metric(

        "Prediction Confidence",

        f"{confidence:.2f}%"
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader(
        "📊 Prediction Probability"
    )


    probability_df = pd.DataFrame({

        "Quality": classes,

        "Probability": probabilities

    })


    probability_df = (
        probability_df
        .set_index("Quality")
    )


    probability_df[
        "Probability"
    ] = (
        probability_df[
            "Probability"
        ] * 100
    )


    st.bar_chart(
        probability_df
    )


    # ========================================================
    # IMAGE FEATURES
    # ========================================================

    st.subheader(
        "👁️ Extracted Image Features"
    )


    image_feature_display = pd.DataFrame({

        "Image Feature": [

            "Brightness",

            "Color Variation",

            "Texture",

            "Edge Density",

            "Crack Score",

            "Defect Score"
        ],

        "Value": [

            image_features[
                "image_brightness"
            ],

            image_features[
                "image_color_variation"
            ],

            image_features[
                "image_texture"
            ],

            image_features[
                "image_edge_density"
            ],

            image_features[
                "image_crack_score"
            ],

            image_features[
                "image_defect_score"
            ]
        ]
    })


    st.dataframe(

        image_feature_display,

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader(
        "📋 Input Summary"
    )


    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write(
            "### 🌳 Wood Parameters"
        )

        st.write(
            f"**Wood Type:** {wood_type}"
        )

        st.write(
            f"**Age:** {age} years"
        )

        st.write(
            f"**Moisture:** {moisture}%"
        )

        st.write(
            f"**Density:** {density}"
        )

        st.write(
            f"**Hardness:** {hardness}"
        )

        st.write(
            f"**Weight:** {weight} kg"
        )

        st.write(
            f"**Thickness:** {thickness} mm"
        )


    with summary_col2:

        st.write(
            "### 🔍 Inspection Parameters"
        )

        st.write(
            f"**Color:** {color}"
        )

        st.write(
            f"**Grain Pattern:** {grain_pattern}"
        )

        st.write(
            f"**Surface:** {surface_condition}"
        )

        st.write(
            f"**Texture:** {texture_input}"
        )

        st.write(
            f"**Defects:** {defects}"
        )

        st.write(
            f"**Crack Level:** {crack_level}"
        )

        st.write(
            f"**Knot Count:** {knot_count}"
        )

        st.write(
            f"**Previous Score:** {previous_score}"
        )

        st.write(
            f"**Durability:** {durability_score}"
        )


    # ========================================================
    # IMAGE ANALYSIS SUMMARY
    # ========================================================

    st.subheader(
        "🖼️ Computer Vision Analysis"
    )


    st.write(
        f"""
        **Brightness:** {image_features["image_brightness"]}

        **Color Variation:** {image_features["image_color_variation"]}

        **Texture:** {image_features["image_texture"]}

        **Edge Density:** {image_features["image_edge_density"]}%

        **Crack Score:** {image_features["image_crack_score"]}

        **Defect Score:** {image_features["image_defect_score"]}
        """
    )


    # ========================================================
    # HOW THE SYSTEM WORKS
    # ========================================================

    st.subheader(
        "🧠 How the AI Prediction Works"
    )


    st.markdown("""
### Step 1 — Image Upload

The user uploads a wood image.

### Step 2 — Image Processing

OpenCV extracts visual features:

- Brightness
- Color variation
- Texture
- Edge density
- Crack indication
- Defect indication

### Step 3 — Wood Parameters

The system receives:

- Wood type
- Age
- Moisture
- Density
- Hardness
- Weight
- Thickness
- Color
- Grain pattern
- Surface condition
- Texture
- Defects
- Crack level
- Knot count
- Previous inspection score
- Durability score

### Step 4 — Random Forest

All features are given to the Random Forest classifier.

### Step 5 — Final Result

The system predicts:

**🟢 GOOD**

**🟡 MEDIUM**

or

**🔴 POOR**
""")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Wood Quality Prediction System | "
    "Python + Streamlit + OpenCV + Scikit-learn + Random Forest"
)

st.caption(
    "Academic prototype. For real-world deployment, "
    "a large labeled wood-image dataset and validation "
    "with domain experts are required."
)