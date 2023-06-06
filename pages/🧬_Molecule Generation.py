import streamlit as st
import ast
import molvs
import rdkit
import pandas as pd
import numpy as np
from PIL import Image
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import AllChem
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from molvs.validate import Validator
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import QED
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import joblib
from streamlit_option_menu import option_menu

from Modeling import calculate_qed
# from Modeling import graph_to_molecule
from Modeling import molecule_from_smiles
from Modeling import smiles_to_graph
from Modeling import DataGenerator
from Modeling import RelationalGraphConvLayer
# from Modeling.VAE import MoleculeGenerator 
import base64

RDLogger.DisableLog("rdApp.*")
st.set_page_config(
    page_title="Geneartion",
    page_icon="🧬",
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

image = Image.open('gen.png')

st.image(image, use_column_width=True)
st.markdown("""<style>
                h1 {
                    color: #304B35;
                }
                </style>""", unsafe_allow_html=True)

st.write("<p style='color: #101913;'>Development of new products often relies on the discovery of novel molecules. While conventional molecular design involves using human expertise to propose, synthesize, and test new molecules, this process can be cost and time intensive, limiting the number of molecules that can be reasonably tested. Generative modeling provides an alternative approach to molecular discovery by reformulating molecular design as an inverse design problem. Here, we review the recent advances in the state-of-the-art of generative molecular design and discusses the considerations for integrating these models into real molecular discovery campaigns.<br></br>Experimental methods for molecule generation involve the synthesis and testing of new compounds in a laboratory setting. This may involve the use of chemical reactions and other techniques to create new compounds, as well as various analytical methods to characterize their properties and behavior.</p>", unsafe_allow_html=True)
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
def inference(model, batch_size):
    z = tf.random.normal((batch_size, LATENT_DIM))
    reconstruction_adjacency, reconstruction_features = model.decoder.predict(z)
    # obtain one-hot encoded adjacency tensor
    adjacency = tf.argmax(reconstruction_adjacency, axis=1)
    adjacency = tf.one_hot(adjacency, depth=BOND_DIM, axis=1)
    # Remove potential self-loops from adjacency
    adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
    # obtain one-hot encoded feature tensor
    features = tf.argmax(reconstruction_features, axis=2)
    features = tf.one_hot(features, depth=ATOM_DIM, axis=2)
    return [
        graph_to_molecule([adjacency[i].numpy(), features[i].numpy()])
        for i in range(batch_size)
    ]
st.markdown(
    """
    <style>
    .custom-file-input::-webkit-file-upload-button {
        background-color: #304B35;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        display: inline-block;
        cursor: pointer;
    }
    .custom-file-input::-webkit-file-upload-button:hover {
        background-color: #3A5A40;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type=['txt'])
if uploaded_file is not None:
    # read the file into a DataFrame
    df = pd.read_csv(uploaded_file, sep='\t', header=None, names=['canonical_smiles'])


st.markdown("""
<style>
    .stButton button {
        background-color: #BAC6A9 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
a=True;
if st.button('Generate Molecule', key='my-button'):
    if uploaded_file is None:
        # Display an error message if no file has been uploaded
        st.error("Please upload a file before processing")
    else:
        #load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)
        # st.markdown('<h1 style="color: #101913;font-size: 28px">Original input data</h1>', unsafe_allow_html=True)
        # hide_table_row_index = """
        #         <style>
        #            table {
        #             border-collapse: collapse !important;
        #             border: 3px solid #94a394 !important;
        #             width: 100%;
        #             margin-top: 5em;
        #             }
        #             th, td {
        #                 border: 2px solid #94a394;
        #                 padding: 8px;
        #                 text-align: left;
        #             }
        #             th {
        #                 background-color: #BAC6A9;
        #             }
        #             th:not(:first-child) {
        #                 border-left: 2px solid #94a394;
        #             }
        #             th:first-child, td:first-child {
        #                 display: none;
        #             }
        #         </style>
        #         """
        # st.markdown(hide_table_row_index, unsafe_allow_html=True)
        # desc_df = df.style.set_properties(**{'color': '#101913'})
        # st.table(desc_df.)

        # Read in calculated descriptors and display the dataframe
        with st.spinner(""):
            df['qed'] = df['canonical_smiles'].apply(calculate_qed)
        # st.header('**df with Calculated qed**')
        # st.write(df)
        # st.write(df.shape)
        # Display SMILES, logP, and qed values for molecule at index 100
        

        # Generate molecule object for SMILES string at index 29
        molecule = molecule_from_smiles(df.iloc[4].canonical_smiles)

        #mol = Chem.MolFromSmiles(df.iloc[3].canonical_smiles)

        # Draw molecule as a 2D image using RDKit
        img = Draw.MolToImage(molecule)

        # Display image in Streamlit
        # st.image(img, caption='Molecule')
        
        # Apply trained model to make prediction on query compounds
        unique_atoms = set()
        for smile in df['canonical_smiles']:
            mol = Chem.MolFromSmiles(smile)
            atoms = mol.GetAtoms()
            for atom in atoms:
                symbol = atom.GetSymbol()
                unique_atoms.add(symbol)
            list_unique_atoms = list(unique_atoms)
        # Print the unique atomic symbols
        list_unique_atoms.append("H")  
        SMILE_CHARSET = "['{}']".format("', '".join(list_unique_atoms))
        # st.write(SMILE_CHARSET)
        bond_mapping = {
            "SINGLE": 0,
            0: Chem.BondType.SINGLE,
            "DOUBLE": 1,
            1: Chem.BondType.DOUBLE,
            "TRIPLE": 2,
            2: Chem.BondType.TRIPLE,
            "AROMATIC": 3,
            3: Chem.BondType.AROMATIC,
        }
        SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

        MAX_MOLSIZE = max(df['canonical_smiles'].str.len())
        SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
        index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
        atom_mapping = dict(SMILE_to_index)
        atom_mapping.update(index_to_SMILE)

        # st.write(MAX_MOLSIZE)

        
        # def generate_molecules(model, num_molecules):
        #     molecules = model.inference(num_molecules)
        #     return molecules
        ##  Generate training set
       
        # def inference(self, batch_size):
        #         z = tf.random.normal((batch_size, LATENT_DIM))
        #         reconstruction_adjacency, reconstruction_features = model.decoder.predict(z)
        #         # obtain one-hot encoded adjacency tensor
        #         adjacency = tf.argmax(reconstruction_adjacency, axis=1)
        #         adjacency = tf.one_hot(adjacency, depth=BOND_DIM, axis=1)
        #         # Remove potential self-loops from adjacency
        #         adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
        #         # obtain one-hot encoded feature tensor
        #         features = tf.argmax(reconstruction_features, axis=2)
        #         features = tf.one_hot(features, depth=ATOM_DIM, axis=2)
        #         return [
        #             graph_to_molecule([adjacency[i].numpy(), features[i].numpy()])
        #             for i in range(batch_size)
        #         ]
        if selected == "Acetylcholinesterase":
                ### Hyperparameters
            BATCH_SIZE = 20
            EPOCHS =1
            VAE_LR = 5e-4
            NUM_ATOMS = MAX_MOLSIZE # Maximum number of atoms

            ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
            BOND_DIM = 3 + 1  # Number of bond types
            LATENT_DIM = 335 
            
            def graph_to_molecule(graph):
        
                # Reference: https://keras.io/examples/generative/wgan-graphs/
                # Unpack graph
                adjacency, features = graph

                # RWMol is a molecule object intended to be edited
                molecule = Chem.RWMol()

                # Remove "no atoms" & atoms with no bonds
                keep_idx = np.where(
                    (np.argmax(features, axis=1) != ATOM_DIM - 1)
                    & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
                )[0]
                features = features[keep_idx]
                adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

                # Add atoms to molecule
                for atom_type_idx in np.argmax(features, axis=1):
                    atom = Chem.Atom(atom_mapping[atom_type_idx])
                    _ = molecule.AddAtom(atom)

                # Add bonds between atoms in molecule; based on the upper triangles
                # of the [symmetric] adjacency tensor
                (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
                for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
                    if atom_i == atom_j or bond_ij == BOND_DIM - 1:
                        continue
                    bond_type = bond_mapping[bond_ij]
                    molecule.AddBond(int(atom_i), int(atom_j), bond_type)

                # Sanitize the molecule; for more information on sanitization, see
                # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
                flag = Chem.SanitizeMol(molecule, catchErrors=True)
                # Let's be strict. If sanitization fails, return None
                if flag != Chem.SanitizeFlags.SANITIZE_NONE:
                    return None

                return molecule
            loaded_model = tf.keras.models.load_model('Vae_Alzheimer')
            #st.write(molecules)
            molecules = inference(loaded_model,1500)
        if selected == "Inflammatory bowel disease":
            ### Hyperparameters
            BATCH_SIZE = 20
            EPOCHS =1
            VAE_LR = 5e-4
            NUM_ATOMS = MAX_MOLSIZE # Maximum number of atoms

            ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
            BOND_DIM = 3 + 1  # Number of bond types
            LATENT_DIM = 335 
            
            def graph_to_molecule(graph):
        
                # Reference: https://keras.io/examples/generative/wgan-graphs/
                # Unpack graph
                adjacency, features = graph

                # RWMol is a molecule object intended to be edited
                molecule = Chem.RWMol()

                # Remove "no atoms" & atoms with no bonds
                keep_idx = np.where(
                    (np.argmax(features, axis=1) != ATOM_DIM - 1)
                    & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
                )[0]
                features = features[keep_idx]
                adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

                # Add atoms to molecule
                for atom_type_idx in np.argmax(features, axis=1):
                    atom = Chem.Atom(atom_mapping[atom_type_idx])
                    _ = molecule.AddAtom(atom)

                # Add bonds between atoms in molecule; based on the upper triangles
                # of the [symmetric] adjacency tensor
                (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
                for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
                    if atom_i == atom_j or bond_ij == BOND_DIM - 1:
                        continue
                    bond_type = bond_mapping[bond_ij]
                    molecule.AddBond(int(atom_i), int(atom_j), bond_type)

                # Sanitize the molecule; for more information on sanitization, see
                # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
                flag = Chem.SanitizeMol(molecule, catchErrors=True)
                # Let's be strict. If sanitization fails, return None
                if flag != Chem.SanitizeFlags.SANITIZE_NONE:
                    return None

                return molecule
            loaded_model = tf.keras.models.load_model('Vae_IBD2')
            #st.write(molecules)
            molecules = inference(loaded_model,1200)
        if selected == "Breast Cancer":
                ### Hyperparameters
            BATCH_SIZE = 20
            EPOCHS =1
            VAE_LR = 5e-4
            NUM_ATOMS = MAX_MOLSIZE # Maximum number of atoms

            ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
            BOND_DIM = 3 + 1  # Number of bond types
            LATENT_DIM = 435 
            
            def graph_to_molecule(graph):
        
                # Reference: https://keras.io/examples/generative/wgan-graphs/
                # Unpack graph
                adjacency, features = graph

                # RWMol is a molecule object intended to be edited
                molecule = Chem.RWMol()

                # Remove "no atoms" & atoms with no bonds
                keep_idx = np.where(
                    (np.argmax(features, axis=1) != ATOM_DIM - 1)
                    & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
                )[0]
                features = features[keep_idx]
                adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

                # Add atoms to molecule
                for atom_type_idx in np.argmax(features, axis=1):
                    atom = Chem.Atom(atom_mapping[atom_type_idx])
                    _ = molecule.AddAtom(atom)

                # Add bonds between atoms in molecule; based on the upper triangles
                # of the [symmetric] adjacency tensor
                (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
                for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
                    if atom_i == atom_j or bond_ij == BOND_DIM - 1:
                        continue
                    bond_type = bond_mapping[bond_ij]
                    molecule.AddBond(int(atom_i), int(atom_j), bond_type)

                # Sanitize the molecule; for more information on sanitization, see
                # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
                flag = Chem.SanitizeMol(molecule, catchErrors=True)
                # Let's be strict. If sanitization fails, return None
                if flag != Chem.SanitizeFlags.SANITIZE_NONE:
                    return None

                return molecule
            loaded_model = tf.keras.models.load_model('Vae_aromatase')
            #st.write(molecules)
            molecules = inference(loaded_model,1500)
        # model = MoleculeGenerator(encoder, 
        #                           decoder,
                                  # MAX_MOLSIZE)

        # model.load_weights('simple_model_weights1')

        # model = tf.keras.models.load_model("my_model2")
         
        
        molecules_valides=[]
        # molecules_smiles=[]
        # unique_molecules_smiles_set
        # unique_molecules_smiles_list
        # unique_molecules
        # unique_smiles
        c=0

        for mol in molecules:
            # Convertir la molécule en objet Mol
            if mol is None: 
                c=c+1
            else:
                mol1 = mol.GetMol()
                mw = Chem.Descriptors.MolWt(mol1)  # Poids moléculaire
                logp = Chem.Descriptors.MolLogP(mol1)  # LogP
                hba = Chem.Descriptors.NumHAcceptors(mol1)  # Nombre d'accepteurs de liaison hydrogène
                hbd = Chem.Descriptors.NumHDonors(mol1)  # Nombre de donneurs de liaison hydrogène

                # Afficher les résultats

                # Vérifier si la molécule satisfait les critères de la règle de Lipinski
                if mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5:
                    molecules_valides.append(mol)
                    # Convert list of RWMol objects to list of string representations
                    molecules_smiles = [Chem.MolToSmiles(mol) for mol in molecules_valides]
                             # Convert list of string representations to set to remove duplicates
                    unique_molecules_smiles_set = set(molecules_smiles)
                    # Convert set back to list of string representations
                    unique_molecules_smiles_list = list(unique_molecules_smiles_set)

                    # Convert list of string representations back to list of RWMol objects
                    unique_molecules = [Chem.MolFromSmiles(smiles) for smiles in unique_molecules_smiles_list]
                    unique_smiles = [Chem.MolToSmiles(mol) for mol in unique_molecules] 

        if molecules_valides==[] :
            st.error("No valid molecule")
        else :

            st.image(
            MolsToGridImage([m for m in unique_molecules if m is not None][:850], molsPerRow=5, subImgSize=(260, 160))  
            )     
        
        # # Generate image using RDKit

        # # # Convert to PIL Image
        # pil_img = Image.fromarray(img)

        # # # Display in Streamlit
        # st.image(pil_img)
        

    
else:    
    a=False;
if not a :
    st.info('Upload input data in the sidebar to start!')