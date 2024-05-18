import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="My Portfolio", page_icon="👨", layout="wide")


with st.sidebar:
    selected = option_menu('My Digital Portfolio',

                           ['🧔🏽‍♂️ About Me',
                            '☠️ Suicidal Detection',
                            '📸 Image Classification',
                            '🛳️ Titanic Survival Prediction',
                            '🏠 House Price Prediction'],
                           menu_icon='🧑‍💻',
                           icons=['🧔🏽‍♂️', '🧔🏽‍♂️','🧔🏽‍♂️','🧔🏽‍♂️','🧔🏽‍♂️'],
                           default_index=0)

#======================================================================================== About ME ============================================================================================================================    


if selected == '🧔🏽‍♂️ About Me':
    # Adding the basic info about me nd the Display Pic
    dp, details = st.columns(2)
    dp.image("./profile-pic.png",width=270)

    with details:
        st.title('Pravesh Singh')
        st.write("A Quick learning individual looking for a potential chance to work in the Machine Learning space")
        with open("./CV.pdf", "rb") as f:
            pdf_data = f.read()
        
        st.download_button(
            label="📄Download Resume", 
            data=pdf_data, 
            file_name="Pravesh's-Resume.pdf", 
            mime="application/octet-stream"
        )
        email_style = """
        <style>
        .email-text {
            color: white; /* Initial color */
            font-weight: bold; /* Initial font weight */
        }
        
        .email-text a:hover {
            color: #ff69b4; /* Color on hover */
        }
        </style>
        """
        
        # Render the email with the defined CSS style
        st.write("📫", '<span class="email-text">praveshds1800@gmail.com</span>', unsafe_allow_html=True)

        # Render the CSS style
        st.markdown(email_style, unsafe_allow_html=True)




    # Adding the links to my social
    link_style = """
    <style>
    .link-text a {
        color: white; /* Initial color */
        text-decoration: none; /* Remove underline */
        font-weight: bold; /* Initial font weight */
        font-size: 20px; /* Initial font size */
    }

    .link-text a:hover {
        color: #ff69b4; /* Color on hover */
        font-size: 20px; /* Font size on hover */
    }
    </style>
    """

    # Render the links with the defined CSS style
    link, git, port, kaggle = st.columns(4, gap='small')

    # Render the LinkedIn link
    link.markdown(f'<div class="link-text"><a href="http://www.linkedin.com/in/praveshsingh1800" target="_blank">LinkedIn</a></div>', unsafe_allow_html=True)

    # Render the Github link
    git.markdown(f'<div class="link-text"><a href="https://github.com/Pravesh1800" target="_blank">Github</a></div>', unsafe_allow_html=True)

    # Render the Portfolio link
    port.markdown(f'<div class="link-text"><a href="https://portfolio-pravesh.streamlit.app/" target="_blank">Portfolio</a></div>', unsafe_allow_html=True)

    # Render the Kaggle link
    kaggle.markdown(f'<div class="link-text"><a href="https://www.kaggle.com/praveshsingh471" target="_blank">Kaggle</a></div>', unsafe_allow_html=True)

    # Render the CSS style
    st.markdown(link_style, unsafe_allow_html=True)
    
    #--- Education ---
    st.write('\n')
    st.markdown(
        """
        <style>
        .heading-line h2 {
            margin-bottom: 0.25rem; /* Adjust the value as needed */
        }
        .heading-line hr {
            margin-top: 0.1rem; /* Adjust the value as needed */
            margin-bottom: 0.5rem; /* Adjust the value as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    import streamlit as st

    # Define the CSS style for the list
    list_style = """
    <style>
    .no-bullets {
        list-style-type: none; /* Remove bullet points */
        padding-left: 0; /* Remove default left padding */
    }

    .no-bullets li {
        margin-bottom: 10px; /* Add some space between list items */
    }
    </style>
    """

    # Render the CSS style
    st.markdown(list_style, unsafe_allow_html=True)

    # Render the heading and horizontal rule together
    st.write('<div class="heading-line"><h1>Education</h1><hr></div>', unsafe_allow_html=True)

    # Render the text in an unordered list format without bullet points
    st.markdown("""
    <ul class="no-bullets">
        <li>🎓 GRADUATION | Bachelor of Computer Applications | GITAM AUG 2021 – APR 2024</li>
        <li>🎒 12th | Science Stream | KENDRIYA VIDYALAYA 1 (NSB VISAKHAPATNAM) | 86%</li>
        <li>🎒 10th | KENDRIYA VIDYALAYA 1 (NSB VISAKHAPATNAM) | 82%</li>
    </ul>
    """, unsafe_allow_html=True)



    # --- EXPERIENCE ---
    st.write('\n')
    st.markdown(
        """
        <style>
        .heading-line h2 {
            margin-bottom: 0.25rem; /* Adjust the value as needed */
        }
        .heading-line hr {
            margin-top: 0.1rem; /* Adjust the value as needed */
            margin-bottom: 0.5rem; /* Adjust the value as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    import streamlit as st

    # Define the CSS style for the list
    list_style = """
    <style>
    .no-bullets {
        list-style-type: none; /* Remove bullet points */
        padding-left: 0; /* Remove default left padding */
    }

    .no-bullets li {
        margin-bottom: 10px; /* Add some space between list items */
    }
    </style>
    """

    # Render the CSS style
    st.markdown(list_style, unsafe_allow_html=True)

    # Render the heading and horizontal rule
    st.write('<div class="heading-line"><h1>Experience</h1><hr></div>', unsafe_allow_html=True)

    # Render the text in an unordered list format without bullet points
    st.markdown("""
    <ul class="no-bullets">
        <li>✔️ Experienced in extracting actionable insights from data</li>
        <li>✔️ Strong hands-on experience and knowledge in Python and Excel</li>
        <li>✔️ Good understanding of Machine Learning models and their respective applications</li>
        <li>✔️ Excellent team-player and displaying strong sense of initiative on tasks</li>
    </ul>
    """, unsafe_allow_html=True)



    # --- SKILLS ---
    st.write('\n')
    st.markdown(
        """
        <style>
        .heading-line h2 {
            margin-bottom: 0.25rem; /* Adjust the value as needed */
        }
        .heading-line hr {
            margin-top: 0.1rem; /* Adjust the value as needed */
            margin-bottom: 0.5rem; /* Adjust the value as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render the heading and horizontal rule together
    st.write('<div class="heading-line"><h1>Hard Skills</h1><hr></div>', unsafe_allow_html=True)
    st.write("""
    👩‍💻 **Programming**: """)
    st.write('''
        - - _Python (Scikit-learn)_
        - - _Python (Tensorflow)_
        - - _Python (Pandas)_ 
        - - _SQL_

    ''')

    st.write("""
    📊 **Data Visulization** """)
    st.write('''
        - - _MS Excel_
        - - _Python(Matplotlib)_
        - - _Python(Seaborn)_ 
    ''')

    st.write("""
    📚 **Modeling** """)
    st.write('''
        - - _Logistic regression_
        - - _Linear regression_
        - - _Decision trees_ 
        - - _Artificial Neural Networks_
    ''')

    st.write("""
    🗄️ **Databases** """)
    st.write('''
        - - _MySQL_
    ''')

    st.write("""
    💻**Web Application** """)
    st.write('''
        - - _Python(Streamlit)_
    ''')


    # --- Projects & Accomplishments ---
    st.write('\n')
    st.markdown(
        """
        <style>
        .heading-line h2 {
            margin-bottom: 0.25rem; /* Adjust the value as needed */
        }
        .heading-line hr {
            margin-top: 0.1rem; /* Adjust the value as needed */
            margin-bottom: 0.5rem; /* Adjust the value as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render the heading and horizontal rule together
    st.write('<div class="heading-line"><h1>Projects & Accomplishments</h1><hr></div>', unsafe_allow_html=True)
    # --- Project 1
    st.write("😞", "**Sucidal Tendency Text Classification | Machine Learning Classification Model**")
    st.write("Mar 2024")
    st.write(
        """
    - ► Applied Support Vector Classification to predict suicidal tendencies in the text data.
    - ► Conducted Exploratory Data Analysis (EDA) to understand data distributions, manage missing values,
        and remove outliers.
    - ► Implemented feature engineering techniques to optimize data representation.
    - ► Achieved a 94% accuracy rate, demonstrating the efficacy of the developed model in identifying
        potential suicidal behavior.
    - ► Created an application using Streamlit and deployed it on the Streamlit cloud.
    """
    )
    # --- Project 2
    st.write("")
    st.write("🏠", "**Banglore House Price Prediction | Machine Learning Regression Model**")
    st.write("Mar 2024")
    st.write(
        """
    - ► Developed this project for the Kaggle house price prediction competition.
    - ► Conducted Exploratory Data Analysis (EDA) to understand data distributions, manage missing values,
        and remove outliers.
    - ► Employed Linear Regression to predict house prices based on dataset from Kaggle.
    - ► Achieved an outstanding accuracy rate of 86%, showcasing the effectiveness of the model.
    - ► Created an application using Streamlit and deployed it on the Streamlit cloud.
    """
    )
    # --- Project 3
    st.write("")
    st.write("🛳️", "**TITANIC SURVIVAL PREDICTION | Machine Learning Classification Model**")
    st.write("Apr 2024")
    st.write(
        """
    - ► Developed this project for the Kaggle Titanic survival prediction competition.
    - ► Conducted Exploratory Data Analysis (EDA) to understand data distributions, manage missing values,
        and remove outliers.
    - ► Utilized the Random Forest Algorithm for efficient prediction.
    - ► Achieved an accuracy rate of 89%, ensuring reliable identification of fraudulent transactions.
    - ► Created an application using Streamlit and deployed it on the Streamlit cloud.
    """
    )




#======================================================================================== Sucidal Detection ============================================================================================================================


if selected == "☠️ Suicidal Detection":
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import one_hot
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from pathlib import Path
    import re
    import zipfile
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()


    st.write(zipfile.is_zipfile(
        "Sucidal_classification/Sucidal_classification.keras"
    ))

    
    if Path("Sucidal_classification/Sucidal_classification.keras").exists() == False :
        st.write("Model might be downloading, please wait.....")
    
    else:
        model = load_model("Sucidal_classification/Sucidal_classification.keras")


        st.title("Sucidal text classification")
        st.image('Sucidal_classification/img.jpg',use_column_width=True)
        st.write("Please enter the text you want to analyze in the text box and you will get the result as suicidal if the text has suicidal tendency or non suicidal if the text has non-suicidal tendency")

        text = st.text_input("Enter the text you want to check")


        voc_size = 5000
        sent_length = 70


        sen = re.sub('[^a-zA-Z]',' ', text)
        sen = sen.lower()
        sen = sen.split()
        sen = [ps.stem(word) for word in sen if not word in stopwords.words('english')]
        sen = ' '.join(sen)
        encoded_sen = [one_hot(sen,voc_size)]
        embeded_sen = pad_sequences(encoded_sen,padding='post',maxlen = sent_length)

        pred = model.predict(embeded_sen)
        posibility = pred.round(4)

        if text :
            if posibility<0.5:
                st.info("This text does not have a sucidal tendency.")
                st.image('Sucidal_classification/non-sucidal.gif', use_column_width=True)

            else:
                st.warning("This text may have sucidal tendency.")
                st.image('Sucidal_classification\sucidal.gif', use_column_width=True)
                st.write(posibility)


#======================================================================================== Image Classification ============================================================================================================================    
if selected == '📸 Image Classification':
    import streamlit as st
    import os
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image_dataset_from_directory
    from PIL import Image
    import numpy as np

    st.title("Multiple Image Uploader")

    # Prompt user for the number of classes
    num_classes = st.number_input("Enter the number of classes", min_value=1, step=1)

    # Building the model
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import ResNet101

    def build_model(num_classes):
        base_model = ResNet101  (weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    def train(model, data):
        status_text = st.empty()  # Create an empty element for dynamic updating
        for epoc in range(epoch):
            model.fit(data, epochs=1)  # Train for 1 epoch at a time
            status_text.write(f"Epoch {epoc + 1} / {epoch} completed")  # Update the status
        #model.save('image_classification.keras')

    def main():
        train_dir = "train_data"
        os.makedirs(train_dir, exist_ok=True)

        for i in range(num_classes):
            st.header(f"Class {i+1}")
            class_name = st.text_input(f"Enter the name of class {i+1}")
            form_key = f"class_{i}_{class_name}_uploader"
            with st.form(key=form_key):
                file_uploader_key = f"class_{i}_{class_name}_uploader"
                uploaded_files = st.file_uploader(f"Choose images for class {class_name}", key=file_uploader_key, accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
                submit_button = st.form_submit_button(label='Submit')
                if submit_button:
                    if uploaded_files:
                        st.write("Sample Images:")
                        num_images = min(len(uploaded_files), 5)
                        num_columns = min(num_images, 5)
                        columns = st.columns(num_columns)
                        for i, uploaded_file in enumerate(uploaded_files[:num_images]):
                            columns[i].image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
                        class_dir = os.path.join(train_dir, class_name)
                        os.makedirs(class_dir, exist_ok=True)
                        for uploaded_file in uploaded_files:
                            with open(os.path.join(class_dir, uploaded_file.name), "wb") as f:
                                f.write(uploaded_file.getbuffer())

        dir = [i for i in os.listdir(train_dir)]
        data = image_dataset_from_directory(train_dir, batch_size=16, image_size=(224, 224), class_names=dir)

        # One-hot encode the labels
        def preprocess_data(x, y):
            x = x / 255.0  # Normalize pixel values
            y = tf.one_hot(y, depth=num_classes)  # One-hot encode labels
            return x, y

        data = data.map(preprocess_data)

        model = build_model(num_classes)

        
        st.header(" Training the Model")
        global epoch
        epoch = st.slider("Enter the no of epochs you want to run.")    


        if st.button("Train Model"):
            train(model, data)

        # Testing
        st.title("Prediction")
        option = st.radio("Select Input Option", ("Upload Image", "Use Camera"))
        if option == "Upload Image":
            uploaded_image = st.file_uploader("Upload an image for prediction", type=['png', 'jpg', 'jpeg'])
        else:
            uploaded_image = st.camera_input("Capture a image to be predicted")

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image = image.resize((224, 224))
            st.image(image, caption='Uploaded Image', use_column_width=True)
            image = np.array(image) / 255.0  # Normalize pixel values
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            prediction = model.predict(image)
            predicted_class_index = np.argmax(prediction)
            st.write(prediction)
            st.write(f"Predicted class: {predicted_class_index}")
            st.write(dir[predicted_class_index])

    if __name__ == "__main__":
        main()



#================================================================================ Titanic Survival Detection ============================================================================================================================

if selected == "🛳️ Titanic Survival Prediction":
    import joblib
    import pandas as pd
    import numpy as np
    import time

    model = joblib.load("./titanic/model.joblib")
    encoder = joblib.load("./titanic/encoder.joblib")
    scaler = joblib.load("./titanic/scaler.joblib")



    st.title("Titanic Survival Prediction")

    st.image("./titanic/img.jpg",use_column_width=True)

    st.write("Please enter all the details")

    Age = st.slider("Enter your age")
    st.write("Your age is :",Age)

    SibSp = st.number_input("Number of siblings / spouses aboard the Titanic",min_value=0)

    Parch = st.number_input("Number of parents / children aboard the Titanic",min_value=0)

    Fare = st.number_input("How much did you pay for the ticket (in $)",min_value=0.0,step=1.0)

    deck = st.slider("Enter your deck",min_value=1,max_value=8)

    Pclass = st.selectbox('Please select the class: 1: Upper , 2: Middle , 3: Lower',[1,2,3])

    Sex = st.selectbox('Please enter your gender',['male','female'])

    Embarked = st.selectbox('Please enter the port on which you embarked C: Cherbourg, Q: Queenstown, S: Southampton',['C','Q','S'])

    Titles = st.selectbox("Please enter your title",['Mr','Miss','Mrs','Master','Royalty','Officer'])



    input_data = pd.DataFrame({'Age': [Age], 'SibSp': [SibSp], 'Parch': [Parch], 'Fare': [Fare], 'deck': [deck],'Pclass': [Pclass], 'Sex': [Sex], 'Embarked': [Embarked], 'Titles': [Titles]})
    cols_test = ['Pclass','Sex','Embarked','Titles']
    encoded_cols_test = encoder.transform(input_data[cols_test])

    encoded_df_test = pd.DataFrame(encoded_cols_test, columns=encoder.get_feature_names_out(cols_test))

    input_data = input_data.drop(columns=cols_test)

    encoded_data_test = pd.concat([input_data, encoded_df_test], axis=1)


    if st.button("Submit"):

        # Scalling the features
        
        X = encoded_data_test.iloc[:,:]
        
        X = scaler.transform(X)
        
        X_flat = np.ravel(X).tolist()
        
        X= X.reshape(1,15)
        
        predic = model.predict(X)
        
        progress = st.progress(0)
        for i in range (100):
            time.sleep(0.01)
            progress.progress(i+1)
            
        if predic == 0:
            st.info("You would not have survived the titanic")
            gif_url = "./titanic/sink_funny.gif" 
            st.image(gif_url, caption='Uh-Ohhh you drowned🥲', use_column_width=True)
        elif predic == 1:
            st.info("You would have survived the Titanic")
            gif = "./titanic/survived.gif"
            st.image(gif, caption='Uh-Ohhh you drowned🥲', use_column_width=True)
            st.balloons()


#================================================================================ House Price Detection ============================================================================================================================

if selected == "🏠 House Price Prediction":
    import pandas as pd
    import numpy as np
    import time
    import joblib


    st.sidebar.markdown(" ### Navigation Bar")


    model = joblib.load("./house/model.joblib")
    encoder = joblib.load("./house/encoder.joblib")

    st.title("House Price Prediction")

    st.image("./house/img.jpg",use_column_width=True)


    st.markdown("## Download the template")
    st.write("Please download the template file and fill it up with all the required data, and please make sure not to leave any column empty")
    st.write("If one or more than one column/columns is/are left empty you will be displayed a list of all the empty columns, please use the list as a reference to fill up all of your empty columns ")

    file ="./house/sample.csv"
    data = pd.read_csv(file)
    st.download_button(
        label='Download the Template', 
        data=data.to_csv(index=False), 
        file_name='data.csv', 
        mime='text/csv'
        )

    st.write("You can refer to this text document to get all the information regarding the columns and fill the columns with the relevant data accordingly")

    file_path = "./house/data_description.txt"
    with open(file_path, 'r') as file:
        data = file.readlines()
    st.write("Data description:")
    st.download_button(label='Download the data description', data='\n'.join(data), file_name='data.txt', mime='text/plain')

    st.write("Once you have read the 'data.txt' and understood all the columns and their values download the 'template.csv' and fill up all the columns.")
    st.write("Each row will contain the data of a house , the no of rows depend on the number of house you want to predict the value of.")

    csv_file = st.file_uploader("Upload the file in CSV fromat",type=['csv'])
    if csv_file is not None:
        test = pd.read_csv(csv_file)
        st.write(test)


    if st.button("Submit"):
        
        # Attempting to fill some of the null values
        
        test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())
        test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])
        test['Utilities']=test['Utilities'].fillna(test['Utilities'].mode()[0])
        test['Alley']=test['Alley'].fillna('none')
        test['SaleType']=test['SaleType'].fillna('Oth')
        test['Exterior1st']=test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
        test['Exterior2nd']=test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0]) 
        test['BsmtFinSF1']=test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mode()[0])    
        test['BsmtFinSF2']=test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mode()[0])
        test['BsmtUnfSF']=test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mode()[0])
        test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mode()[0])
        test['BsmtFullBath']=test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])
        test['BsmtHalfBath']=test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])
        test['KitchenQual']=test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
        test['Functional']=test['Functional'].fillna(test['Functional'].mode()[0])
        test['GarageCars']=test['GarageCars'].fillna(test['GarageCars'].mode()[0])    
        test['MasVnrType']=test['MasVnrType'].fillna('None')
        test['MasVnrArea']=test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
        test['GarageArea']=test['GarageArea'].fillna(test['GarageArea'].mode()[0])    
        test['BsmtQual']=test['BsmtQual'].fillna('No Basement')
        test['BsmtCond']=test['BsmtCond'].fillna('No Basement')
        test['BsmtExposure']=test['BsmtExposure'].fillna('No Basement')
        test['BsmtFinType1']=test['BsmtFinType1'].fillna('No Basement')
        test['BsmtFinType2']=test['BsmtFinType2'].fillna('No Basement')
        test['Electrical']=test['Electrical'].fillna(test['Electrical'].mode()[0])
        test['GarageType']=test['GarageType'].fillna('No Garage')
        test['GarageQual']=test['GarageQual'].fillna('No Garage')
        test['GarageCond']=test['GarageCond'].fillna('No Garage')
        test['FireplaceQu']=test['FireplaceQu'].fillna('No Fireplace')    
        test['PoolQC']=test['PoolQC'].fillna('None')    
        test['Fence']=test['Fence'].fillna('None')    
        test['MiscFeature']=test['MiscFeature'].fillna('None')
        
        # Displaying all the columns that have null value
        null_mask = test.isnull().any()
        columns_with_null = test.columns[null_mask]
        st.write("Columns with null values are:")
        st.write(columns_with_null)
        
        # Encoding the columns
        object_col = test.select_dtypes(include=['object'])
        cols = object_col.columns
        new_test = test
        encoded_cols_test = encoder.transform(object_col)
        encoded_df_test = pd.DataFrame(encoded_cols_test, columns=encoder.get_feature_names_out(cols))
        new_test = new_test.drop(columns=cols)
        encoded_data_test = pd.concat([new_test, encoded_df_test], axis=1)
        
        # Predicting the data
        test_pred = model.predict(encoded_data_test)
        output = pd.DataFrame({'Id': test.Id, 'SalePrice': test_pred})
        st.write("Your Prediction is ready")
        progress = st.progress(0)
        for i in range (100):
            time.sleep(0.01)
            progress.progress(i+1)
        st.write(output)
        st.download_button(
            label='Download your predictions', 
            data=output.to_csv(index=False), 
            file_name='output.csv', 
            mime='text/csv')
