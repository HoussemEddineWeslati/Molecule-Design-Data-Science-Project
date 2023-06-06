import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from streamlit_option_menu import option_menu
from sklearn.feature_selection import VarianceThreshold
st.set_page_config(
    page_icon="🧪",
)
with open("logo.png", "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-75%;margin-left:-2%;">
            <img src="data:image/png;base64,{data}" width="250" height="80">
        </div>
        """,
        unsafe_allow_html=True,
    )


# Logo image
image2 = Image.open('pred.png')
image = Image.open('image2.png')
resized_image = image2.resize((800,120))
st.image(resized_image)
st.image(image, use_column_width=True)

# Page title
st.markdown(
    f'<div style="text-align: left;color: #101913;">Bioactivity of molecules refers to their ability to interact with biological targets such as receptors, enzymes or proteins, and produce a biological response. The biological response can be either beneficial or detrimental to the organism, depending on the type of molecule and its mode of action.<br></br>In drug discovery, the bioactivity of molecules is critical for identifying potential drug candidates. A molecule with a specific bioactivity can be targeted to interact with a particular biological target, thereby producing a therapeutic effect. For example, a molecule that inhibits the activity of a specific receptor can be used to treat a disease associated with overactivity of that receptor.  <br></br>The ability of a molecule to interact with biological targets depends on several factors such as its chemical structure, size, shape, and charge distribution. These factors determine how well a molecule can bind to a biological target and produce a biological response.<br></br> </div>',
    unsafe_allow_html=True)
selected = option_menu(
    menu_title=None,  # requiredIn drug discovery, the bioactivity of molecules is critical for identifying potential drug candidates. A molecule with a specific bioactivity can be targeted to interact with a particular biological target, thereby producing a therapeutic effect. For example, a molecule that inhibits the activity of a specific receptor can be used to treat a disease associated with overactivity of that receptor.  <br></br>The ability of a molecule to interact with biological targets depends on several factors such as its chemical structure, size, shape, and charge distribution. These factors determine how well a molecule can bind to a biological target and produce a biological response.<br></br>
    options=["Acetylcholinesterase","Inflammatory bowel disease","Breast Cancer"],  # required
    icons=["💊", "💊","💊"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
            "container": {"padding": "0!important", "background-color": "#BAC6A9"},
            "icon": {"color": "black", "font-size": "14px"},
            "nav-link": {
                "font-size": "12.5px",
                "text-align": "left",
                "margin": "0px",
                
            },
            "nav-link-selected": {"background-color": "101913"},
        },
    )


if selected == "Acetylcholinesterase":
    st.markdown("""
    <style>
        .stButton button {
            background-color: #BAC6A9 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

        # Model building
    def remove_low_variance(input_data, threshold=0.1):
        selection = VarianceThreshold(threshold)
        selection.fit(input_data)
        return input_data[input_data.columns[selection.get_support(indices=True)]]

    def build_model(input_data):
        # Reads in saved regression model
        load_model = pickle.load(open('acetylcholinesterase_model1.pkl', 'rb'))
        # Apply model to make predictions
        prediction = load_model.predict(input_data)
        st.markdown('<h1 style="color: #101913;font-size: 28px">Prediction output</h1>', unsafe_allow_html=True)
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        desc_df = df.style.set_properties(**{'color': '#101913'})
        hide_table_row_index = """
                <style>
                    table {
                        border-collapse: collapse !important;
                        border: 3px solid #94a394 !important;
                        width: 100%;
                        margin-top: 5em;
                        }
                        th, td {
                            border: 2px solid #94a394;
                            padding: 8px;
                            text-align: left;
                        }
                        th {
                            background-color: #BAC6A9;
                        }
                        th:not(:first-child) {
                            border-left: 2px solid #94a394;
                        }
                        th:first-child, td:first-child {
                            display: none;
                        }
                </style>
                """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Display a static table
        st.table(desc_df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    def build_model2(input_data):
        # Reads in saved regression model
        load_model = pickle.load(open('acetylcholinesterase_model2.pkl', 'rb'))
        # Apply model to make predictions
        prediction = load_model.predict(input_data)
        st.markdown('<h1 style="color: #101913;font-size: 28px">Prediction output</h1>', unsafe_allow_html=True)
        prediction_output = pd.Series(prediction, name='Class')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        desc_df = df.style.set_properties(**{'color': '#101913'})

        hide_table_row_index = """
                <style>
                   table {
                    border-collapse: collapse !important;
                    border: 3px solid #94a394 !important;
                    width: 100%;
                    margin-top: 5em;
                    }
                    th, td {
                        border: 2px solid #94a394;
                        padding: 8px;
                        text-align: left;
                    }
                    th {
                        background-color: #BAC6A9;
                    }
                    th:not(:first-child) {
                        border-left: 2px solid #94a394;
                    }
                    th:first-child, td:first-child {
                        display: none;
                    }
                </style>
                """

                # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        
        # Display a static table
        st.table(desc_df)
        st.markdown(filedownload(df), unsafe_allow_html=True)


 

    st.markdown(
    f'<div style="text-align: left;font-size:20px ;color: #101913;">Bioactivity Prediction for Acetylcholinesterase </div>',
    unsafe_allow_html=True)
    # Sidebar
    
    a=True;
    b=True;
    uploaded_file = st.file_uploader("", type=['txt'])

    col1, col2 = st.columns(2)

    with col1:  
        if st.button('Predict_pIC50', key='my-button'):
            if uploaded_file is None:
                # Display an error message if no file has been uploaded
                st.error("Please upload a file before processing")
            else:
                load_data = pd.read_table(uploaded_file, sep=' ', header=None)
                load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

                st.markdown('<h1 style="color: #101913;font-size: 28px">Original input data</h1>', unsafe_allow_html=True)
                desc_table = load_data.style.set_properties(**{'color': '#101913'})
                hide_table_row_index = """
                <style>
                table {
                border-collapse: collapse !important;
                border: 3px solid #94a394 !important;
                width: 100%;
                margin-top: 5em;
                }
                th, td {
                    border: 2px solid #94a394;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #BAC6A9;
                }
                th:not(:first-child) {
                    border-left: 2px solid #94a394;
                }
                th:first-child, td:first-child {
                    display: none;
                }
                </style>
                """

                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)

                # Display a static table
                st.table(desc_table)
                with st.spinner(""):
                   desc_calc()

              # Read in calculated descriptors and display the dataframe
                # st.header('**Calculated molecular descriptors**')

                desc = pd.read_csv('descriptors_output.csv')
                # st.write(desc)
                # st.write(desc.shape)

                # Read descriptor list used in previously built model
                # st.header('**Subset of descriptors from previously built models**')
                Xlist = list(pd.read_csv('descriptor_list.csv').columns)
                desc_subset = desc[Xlist]
                # desc_subset = remove_low_variance(desc, threshold=0.1)
                # st.write(desc_subset)
                # st.write(desc_subset.shape)
                # Apply trained model to make prediction on query compounds
                build_model(desc_subset)
        else:    
            a=False;

    with col2:
        if st.button('Predict_Bioactivity'):
            if uploaded_file is None:
                # Display an error message if no file has been uploaded
                st.error("Please upload a file before processing")
            else:
                load_data = pd.read_table(uploaded_file, sep=' ', header=None)

                load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

                st.markdown('<h1 style="color: #101913;font-size: 28px">Original input data</h1>', unsafe_allow_html=True)

                # Write the table with the defined style
                desc_table = load_data.style.set_properties(**{'color': '#101913'})
                hide_table_row_index = """
                <style>
                table {
                border-collapse: collapse !important;
                border: 3px solid #94a394 !important;
                width: 100%;
                margin-top: 5em;
                }
                th, td {
                    border: 2px solid #94a394;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #BAC6A9;
                }
                 th:not(:first-child) {
                    border-left: 2px solid #94a394;
                }
                th:first-child, td:first-child {
                    display: none;
                }
                </style>
                """

                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)

                # Display a static table
                st.table(desc_table)
                # st.write(desc_table)
                # st.write(load_data)
                # col1, col2 = st.columns(2)

                with st.spinner(""):
                    desc_calc()

                # Read in calculated descriptors and display the dataframe
                # st.markdown('<h1 style="color: #101913;font-size: 28px">Calculated molecular descriptors</h1>', unsafe_allow_html=True)
                desc = pd.read_csv('descriptors_output.csv')
                # desc_table2 = desc.style.set_properties(**{'color': '#101913'}).set_table_styles([{'selector': 'th', 'props': 'border-left: 1px solid white'}])
                # st.write(desc_table2)
                # st.write(desc.shape)

                # Read descriptor list used in previously built model
                # st.header('**Subset of descriptors from previously built models**')
                Xlist = list(pd.read_csv('descriptor_list.csv').columns)
                desc_subset = desc[Xlist]
                # st.write(desc_subset.shape)

                # Apply trained model to make prediction on query compounds
                build_model2(desc_subset)

        else:
            b=False;
    if not a and not b:
        st.info('Upload input data in the sidebar to start!')
    


if selected == "Inflammatory bowel disease":
    st.markdown("""
    <style>
        .stButton button {
            background-color: #BAC6A9 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

        # Model building
    def remove_low_variance(input_data, threshold=0.1):
        selection = VarianceThreshold(threshold)
        selection.fit(input_data)
        return input_data[input_data.columns[selection.get_support(indices=True)]]

    def build_model(input_data):
        # Reads in saved regression model
        load_model = pickle.load(open('IBD_model1.pkl', 'rb'))
        # Apply model to make predictions
        prediction = load_model.predict(input_data)
        st.markdown('<h1 style="color: #101913;font-size: 28px">Prediction output</h1>', unsafe_allow_html=True)
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        desc_df = df.style.set_properties(**{'color': '#101913'})
        hide_table_row_index = """
                <style>
                    table {
                        border-collapse: collapse !important;
                        border: 3px solid #94a394 !important;
                        width: 100%;
                        margin-top: 5em;
                        }
                        th, td {
                            border: 2px solid #94a394;
                            padding: 8px;
                            text-align: left;
                        }
                        th {
                            background-color: #BAC6A9;
                        }
                        th:not(:first-child) {
                            border-left: 2px solid #94a394;
                        }
                        th:first-child, td:first-child {
                            display: none;
                        }
                </style>
                """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Display a static table
        st.table(desc_df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    def build_model2(input_data):
        # Reads in saved regression model
        load_model = pickle.load(open('IBD_model2.pkl', 'rb'))
        # Apply model to make predictions
        prediction = load_model.predict(input_data)
        st.markdown('<h1 style="color: #101913;font-size: 28px">Prediction output</h1>', unsafe_allow_html=True)
        prediction_output = pd.Series(prediction, name='Class')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        desc_df = df.style.set_properties(**{'color': '#101913'})

        hide_table_row_index = """
                <style>
                   table {
                    border-collapse: collapse !important;
                    border: 3px solid #94a394 !important;
                    width: 100%;
                    margin-top: 5em;
                    }
                    th, td {
                        border: 2px solid #94a394;
                        padding: 8px;
                        text-align: left;
                    }
                    th {
                        background-color: #BAC6A9;
                    }
                    th:not(:first-child) {
                        border-left: 2px solid #94a394;
                    }
                    th:first-child, td:first-child {
                        display: none;
                    }
                </style>
                """

                # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        
        # Display a static table
        st.table(desc_df)
        st.markdown(filedownload(df), unsafe_allow_html=True)


 

    st.markdown(
    f'<div style="text-align: left;font-size:20px ;color: #101913;">Bioactivity Prediction for Inflammatory bowel disease </div>',
    unsafe_allow_html=True)
    # Sidebar
    
    a=True;
    b=True;
    uploaded_file = st.file_uploader("", type=['txt'])

    col1, col2 = st.columns(2)

    with col1:  
        if st.button('Predict_pIC50', key='my-button'):
            if uploaded_file is None:
                # Display an error message if no file has been uploaded
                st.error("Please upload a file before processing")
            else:
                load_data = pd.read_table(uploaded_file, sep=' ', header=None)
                load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

                st.markdown('<h1 style="color: #101913;font-size: 28px">Original input data</h1>', unsafe_allow_html=True)
                desc_table = load_data.style.set_properties(**{'color': '#101913'})
                hide_table_row_index = """
                <style>
                table {
                border-collapse: collapse !important;
                border: 3px solid #94a394 !important;
                width: 100%;
                margin-top: 5em;
                }
                th, td {
                    border: 2px solid #94a394;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #BAC6A9;
                }
                th:not(:first-child) {
                    border-left: 2px solid #94a394;
                }
                th:first-child, td:first-child {
                    display: none;
                }
                </style>
                """

                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)

                # Display a static table
                st.table(desc_table)
                with st.spinner(""):
                   desc_calc()

              # Read in calculated descriptors and display the dataframe
                # st.header('**Calculated molecular descriptors**')

                desc = pd.read_csv('descriptors_output2.csv')
                # st.write(desc)
                # st.write(desc.shape)

                # Read descriptor list used in previously built model
                # st.header('**Subset of descriptors from previously built models**')
                Xlist = list(pd.read_csv('descriptor_list3.csv').columns)
                desc_subset = desc[Xlist]
                # desc_subset = remove_low_variance(desc, threshold=0.1)
                # st.write(desc_subset)
                # st.write(desc_subset.shape)
                # Apply trained model to make prediction on query compounds
                build_model(desc_subset)
        else:    
            a=False;

    with col2:
        if st.button('Predict_Bioactivity'):
            if uploaded_file is None:
                # Display an error message if no file has been uploaded
                st.error("Please upload a file before processing")
            else:
                load_data = pd.read_table(uploaded_file, sep=' ', header=None)

                load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

                st.markdown('<h1 style="color: #101913;font-size: 28px">Original input data</h1>', unsafe_allow_html=True)

                # Write the table with the defined style
                desc_table = load_data.style.set_properties(**{'color': '#101913'})
                hide_table_row_index = """
                <style>
                table {
                border-collapse: collapse !important;
                border: 3px solid #94a394 !important;
                width: 100%;
                margin-top: 5em;
                }
                th, td {
                    border: 2px solid #94a394;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #BAC6A9;
                }
                 th:not(:first-child) {
                    border-left: 2px solid #94a394;
                }
                th:first-child, td:first-child {
                    display: none;
                }
                </style>
                """

                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)

                # Display a static table
                st.table(desc_table)
                # st.write(desc_table)
                # st.write(load_data)
                # col1, col2 = st.columns(2)

                with st.spinner(""):
                    desc_calc()

                # Read in calculated descriptors and display the dataframe
                # st.markdown('<h1 style="color: #101913;font-size: 28px">Calculated molecular descriptors</h1>', unsafe_allow_html=True)
                desc = pd.read_csv('descriptors_output2.csv')
                # desc_table2 = desc.style.set_properties(**{'color': '#101913'}).set_table_styles([{'selector': 'th', 'props': 'border-left: 1px solid white'}])
                # st.write(desc_table2)
                # st.write(desc.shape)

                # Read descriptor list used in previously built model
                # st.header('**Subset of descriptors from previously built models**')
                Xlist = list(pd.read_csv('descriptor_list3.csv').columns)
                desc_subset = desc[Xlist]
                # st.write(desc_subset.shape)

                # Apply trained model to make prediction on query compounds
                build_model2(desc_subset)

        else:
            b=False;
    if not a and not b:
        st.info('Upload input data in the sidebar to start!')
    

  

if selected == "Breast Cancer":
    st.markdown("""
    <style>
        .stButton button {
            background-color: #BAC6A9 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

        # Model building
    def remove_low_variance(input_data, threshold=0.1):
        selection = VarianceThreshold(threshold)
        selection.fit(input_data)
        return input_data[input_data.columns[selection.get_support(indices=True)]]

    def build_model(input_data):
        # Reads in saved regression model
        load_model = pickle.load(open('aromatase_model1.pkl', 'rb'))
        # Apply model to make predictions
        prediction = load_model.predict(input_data)
        st.markdown('<h1 style="color: #101913;font-size: 28px">Prediction output</h1>', unsafe_allow_html=True)
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        desc_df = df.style.set_properties(**{'color': '#101913'})
        hide_table_row_index = """
                <style>
                    table {
                        border-collapse: collapse !important;
                        border: 3px solid #94a394 !important;
                        width: 100%;
                        margin-top: 5em;
                        }
                        th, td {
                            border: 2px solid #94a394;
                            padding: 8px;
                            text-align: left;
                        }
                        th {
                            background-color: #BAC6A9;
                        }
                        th:not(:first-child) {
                            border-left: 2px solid #94a394;
                        }
                        th:first-child, td:first-child {
                            display: none;
                        }
                </style>
                """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Display a static table
        st.table(desc_df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    def build_model2(input_data):
        # Reads in saved regression model
        load_model = pickle.load(open('aromatase_model2.pkl', 'rb'))
        # Apply model to make predictions
        prediction = load_model.predict(input_data)
        st.markdown('<h1 style="color: #101913;font-size: 28px">Prediction output</h1>', unsafe_allow_html=True)
        prediction_output = pd.Series(prediction, name='Class')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        desc_df = df.style.set_properties(**{'color': '#101913'})

        hide_table_row_index = """
                <style>
                   table {
                    border-collapse: collapse !important;
                    border: 3px solid #94a394 !important;
                    width: 100%;
                    margin-top: 5em;
                    }
                    th, td {
                        border: 2px solid #94a394;
                        padding: 8px;
                        text-align: left;
                    }
                    th {
                        background-color: #BAC6A9;
                    }
                    th:not(:first-child) {
                        border-left: 2px solid #94a394;
                    }
                    th:first-child, td:first-child {
                        display: none;
                    }
                </style>
                """

                # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        
        # Display a static table
        st.table(desc_df)
        st.markdown(filedownload(df), unsafe_allow_html=True)


 

    st.markdown(
    f'<div style="text-align: left;font-size:20px ;color: #101913;">Bioactivity Prediction for Aromatase </div>',
    unsafe_allow_html=True)
    # Sidebar
    
    a=True;
    b=True;
    uploaded_file = st.file_uploader("", type=['txt'])

    col1, col2 = st.columns(2)

    with col1:  
        if st.button('Predict_pIC50', key='my-button'):
            if uploaded_file is None:
                # Display an error message if no file has been uploaded
                st.error("Please upload a file before processing")
            else:
                load_data = pd.read_table(uploaded_file, sep=' ', header=None)
                load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

                st.markdown('<h1 style="color: #101913;font-size: 28px">Original input data</h1>', unsafe_allow_html=True)
                desc_table = load_data.style.set_properties(**{'color': '#101913'})
                hide_table_row_index = """
                <style>
                table {
                border-collapse: collapse !important;
                border: 3px solid #94a394 !important;
                width: 100%;
                margin-top: 5em;
                }
                th, td {
                    border: 2px solid #94a394;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #BAC6A9;
                }
                th:not(:first-child) {
                    border-left: 2px solid #94a394;
                }
                th:first-child, td:first-child {
                    display: none;
                }
                </style>
                """

                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)

                # Display a static table
                st.table(desc_table)
                with st.spinner(""):
                   desc_calc()

              # Read in calculated descriptors and display the dataframe
                # st.header('**Calculated molecular descriptors**')

                desc = pd.read_csv('descriptors_output.csv')
                # st.write(desc)
                # st.write(desc.shape)

                # Read descriptor list used in previously built model
                # st.header('**Subset of descriptors from previously built models**')
                Xlist = list(pd.read_csv('descriptor_list4.csv').columns)
                desc_subset = desc[Xlist]
                # st.write(desc_subset)
                # st.write(desc_subset.shape)
                # Apply trained model to make prediction on query compounds
                build_model(desc_subset)
        else:    
            a=False;

    with col2:
        if st.button('Predict_Bioactivity'):
            if uploaded_file is None:
                # Display an error message if no file has been uploaded
                st.error("Please upload a file before processing")
            else:
                load_data = pd.read_table(uploaded_file, sep=' ', header=None)

                load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

                st.markdown('<h1 style="color: #101913;font-size: 28px">Original input data</h1>', unsafe_allow_html=True)

                # Write the table with the defined style
                desc_table = load_data.style.set_properties(**{'color': '#101913'})
                hide_table_row_index = """
                <style>
                table {
                border-collapse: collapse !important;
                border: 3px solid #94a394 !important;
                width: 100%;
                margin-top: 5em;
                }
                th, td {
                    border: 2px solid #94a394;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #BAC6A9;
                }
                 th:not(:first-child) {
                    border-left: 2px solid #94a394;
                }
                th:first-child, td:first-child {
                    display: none;
                }
                </style>
                """

                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)

                # Display a static table
                st.table(desc_table)
                # st.write(desc_table)
                # st.write(load_data)
                # col1, col2 = st.columns(2)

                with st.spinner(""):
                    desc_calc()

                # Read in calculated descriptors and display the dataframe
                # st.markdown('<h1 style="color: #101913;font-size: 28px">Calculated molecular descriptors</h1>', unsafe_allow_html=True)
                desc = pd.read_csv('descriptors_output.csv')
                # desc_table2 = desc.style.set_properties(**{'color': '#101913'}).set_table_styles([{'selector': 'th', 'props': 'border-left: 1px solid white'}])
                # st.write(desc_table2)
                # st.write(desc.shape)

                # Read descriptor list used in previously built model
                # st.header('**Subset of descriptors from previously built models**')
                Xlist = list(pd.read_csv('descriptor_list4.csv').columns)
                desc_subset = desc[Xlist]
                # st.write(desc_subset.shape)

                # Apply trained model to make prediction on query compounds
                build_model2(desc_subset)

        else:
            b=False;
    if not a and not b:
        st.info('Upload input data in the sidebar to start!')
    