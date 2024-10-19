# This is app is a modified version of the app created by Chanin Nantasenamat (Data Professor) https://youtube.com/dataprofessor
# Credit: This app is inspired by https://huggingface.co/spaces/osanseviero/esmfold

import requests
import json
import time

import numpy as np
import pandas as pd

import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
# For molstar visualization
from streamlit_molstar import st_molstar

from Bio.SeqUtils.ProtParam import ProteinAnalysis

USE_STMOL = False

pspired_uri = 'http://bioinf.cs.ucl.ac.uk/psipred/api/'
esm_uri = 'https://api.esmatlas.com/foldSequence/v1/pdb/'

#st.set_page_config(layout = 'wide')
st.sidebar.title('🎈 ESMFold')
st.sidebar.write('[*ESMFold*](https://esmatlas.com/about) is an end-to-end single sequence protein structure predictor based on the ESM-2 language model. For more information, read the [research article](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2) and the [news article](https://www.nature.com/articles/d41586-022-03539-1) published in *Nature*.')

# pdb
def render_mol_molstar(pdb_file):
    st_molstar(pdb_file, height="480px")

# stmol
def render_mol_stmol(pdb):
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb,'pdb')
    pdbview.setStyle({'cartoon':{'color':'spectrum'}})
    pdbview.setBackgroundColor('white')#('0xeeeeee')
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(True)
    showmol(pdbview, height = 500,width=800)

def parse_horiz_data(horiz_data):
    # Split the data into lines
    lines = horiz_data.strip().split('\n')
    
    # Initialize lists to store amino acids and corresponding structures
    amino_acids = ["Amino Acids"]
    structures = ["Predcited Structures"]
    
    # Iterate over lines
    for line in lines:
        if line.startswith("  AA:"):
            # Extract amino acids from the line
            amino_acids.extend(list(line.split()[1]))
        elif line.startswith("Pred:"):
            # Extract secondary structures from the line
            structures.extend(list(line.split()[1]))
    
    # Combine amino acids and structures into pairs
    pairs = [[aa, ss] for aa, ss in zip(amino_acids, structures)]
    
    # Convert the pairs into a numpy array
    return np.array(pairs)

# Physicochemical parameter prediction

def phy_param_pred(fasta_seq:str):
    name = fasta_seq.split("\n")[0][1:]
    sequence = fasta_seq.split("\n")[1]

    analysis = ProteinAnalysis(sequence)

    st.write("Total amino acids present: " + str(sum(analysis.count_amino_acids().values())))

    with st.expander("Amino Acids Composition"):
        a = analysis.count_amino_acids()
        b = analysis.get_amino_acids_percent()
        table = []
        for aa in a:
            table.append([aa, a[aa], f"{b[aa]*100}%"])
        table = np.array(table)
        st.dataframe(pd.DataFrame(table, columns=["Amino Acid", "Quantity", "Perecentage"]))

    with st.expander("Other Attributes"):
        st.info("The seconday strcuture composition may differ from the predicted composition by s4pred.")
        sec_comp = analysis.secondary_structure_fraction()
        table = [
            ['Molecular Weight', analysis.molecular_weight()],
            ['Aromacity', analysis.aromaticity()],
            ['Instability Index', analysis.instability_index()],
            ['Helix Percentage', f"{sec_comp[0]*100}%"],
            ['Turn Percentage', f"{sec_comp[1]*100}%"],
            ['Sheet Percentage', f"{sec_comp[2]*100}%"],
            ['Molar Extinction Coefficient with reduced cysteines', analysis.molar_extinction_coefficient()[0]],
            ['Molar Extinction Coefficient with disulfid bridges', analysis.molar_extinction_coefficient()[1]]
        ]
        st.dataframe(pd.DataFrame(table, columns=["Attribute", "Value"]))

#s4pred
def s4pred_pred(fasta_seq:str):
    name = fasta_seq.split("\n")[0][1:]
    payload = {'input_data': fasta_seq.split("\n")[1]}
    data = {'job': 's4pred',
        'submission_name': f'prediction_{name}',
        'email': 'na@johndoe.com',
        }
    with st.status('Sending seqeunce to s4pred...', expanded=False) as status:
        r = requests.post(pspired_uri+"submission"+".json", data=data, files=payload)
        response_data = json.loads(r.text)
        print(response_data)
        while True:
            if 'error' in response_data:
                status.update(label=f"Failed to get the prediction: {response_data['error']}", state="error")
                break
            
            status.update(label="Polling result for: "+name)
            result_uri = pspired_uri+"submission/"+response_data['UUID']
            r = requests.get(result_uri, headers={"Accept":"application/json"})
            result_data = json.loads(r.text)

            if "Complete" in result_data["state"]:
                status.update(label="Completed: "+name,state="complete")

                data_path_uri = pspired_uri+result_data['submissions'][0]['results'][2]['data_path']
                r = requests.get(data_path_uri, headers={"Accept":"text/plain"})
                st.text(".horiz format")
                ss_pred = parse_horiz_data(r.text)
                with st.container(height=300):
                    st.code(r.text)
                st.dataframe(ss_pred.transpose())
                st.link_button(
                    label=f"Download horiz: {name}",
                    url=data_path_uri
                )

                data_path_uri = pspired_uri+result_data['submissions'][0]['results'][1]['data_path']
                r = requests.get(data_path_uri, headers={"Accept":"text/plain"})
                st.text(".ss2 format")
                with st.container(height=300):
                    st.code(r.text)
                st.link_button(
                    label=f"Download ss2: {name}",
                    url=data_path_uri
                )

                break
            elif "Error" in result_data["state"]:
                status.update(label="Failed to get the prediction", state="error")
                break
            else:
                time.sleep(30)
    return ss_pred

# ESMfold
def parse_fasta(fasta):
    sequences = []
    current_sequence = ""
    lines = fasta.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            # New sequence header found
            if current_sequence:
                valid = True
                for i in current_sequence.split("\n")[1]:
                    if i not in {'Y', 'V', 'K', 'N', 'G', 'W', 'Q', 'I', 'L', 'Z', 'F', 'H', 'A', 'D', 'P', 'R', 'S', 'E', 'B', 'X', 'J', 'C', 'T', 'M'}:
                        valid = False
                if valid:
                    sequences.append(current_sequence)
                else:
                    s_name = {current_sequence.split("\n")[0][1:]}
                    st.error(f'Sequence: {s_name} is invalid!', icon="🚨")
            current_sequence = line + "\n"
        else:
            # Append sequence data
            current_sequence += line
    # Append the last sequence
    if current_sequence:
        valid = True
        for i in current_sequence.split("\n")[1]:
            if i not in {'Y', 'V', 'K', 'N', 'G', 'W', 'Q', 'I', 'L', 'Z', 'F', 'H', 'A', 'D', 'P', 'R', 'S', 'E', 'B', 'X', 'J', 'C', 'T', 'M'}:
                valid = False
        if valid:
            sequences.append(current_sequence)
        else:
            s_name = {current_sequence.split("\n")[0][1:]}
            st.error(f'Sequence: {s_name} is invalid!', icon="🚨")
    return sequences

def update(sequence):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    sequences = parse_fasta(sequence)
    pdb_strings = []

    for idx, seq in enumerate(sequences):
        response = requests.post(esm_uri, headers=headers, data=seq, verify=False)
        pdb_string = response.content.decode('utf-8')
        pdb_strings.append(pdb_string)

    for idx, pdb_string in enumerate(pdb_strings):
        with open(f'predicted_{idx}.pdb', 'w') as f:
            f.write(pdb_string)
        try:
            struct = bsio.load_structure(f'predicted_{idx}.pdb', extra_fields=["b_factor"])
        except:
            continue
        b_value = round(struct.b_factor.mean(), 4)

        # Display protein structure
        st.subheader(f'Visualization of predicted protein structure {idx+1}')
        
        if USE_STMOL:
            render_mol_stmol(pdb_string)
        else:
            render_mol_molstar(f'predicted_{idx}.pdb')
        # plDDT value is stored in the B-factor field
        st.subheader(f'plDDT {idx+1}')
        st.write('plDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100.')
        st.info(f'plDDT: {b_value}')

        st.download_button(
            label=f"Download PDB {idx+1}",
            data=pdb_string,
            file_name=f'predicted_{idx}.pdb',
            mime='text/plain',
            key=f"download_button_{idx}"  # Unique key for each download button
        )

        # Physicochemical parameter prediction
        st.subheader("Physicochemical parameter prediction")
        phy_param_pred(sequences[idx])

        # Get s4pred output
        st.subheader("s4pred - Seconday Structure prediction")
        ss_pred = s4pred_pred(sequences[idx])

        st.divider()

# Protein sequence input
DEFAULT_SEQ = ">Sequence_1\nMGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ\n>Sequence_2\nYOUR_NEXT_SEQUENCE_HERE\n>Sequence_3\nANOTHER_SEQUENCE_HERE"
sequence_input = st.sidebar.text_area('Input sequence(s) in FASTA format', DEFAULT_SEQ, height=275)

predict = st.sidebar.button('Batch Predict', on_click=lambda: update(sequence_input), key="predict_button")  # Unique key for the Predict button

if not predict:
    st.warning('👈 Enter protein sequence data!')