import argparse
import pandas as pd
import os
from tqdm import tqdm

def main(input_folder, output_folder = None):

    # Create storage
    gene_annotations = None
    gene_counts_tpm = []
#     gene_counts_fpkm = []
#     gene_counts_fpkm_uq = []

    # Load patients list
    df_patients = None
    for file in os.listdir(input_folder):
        if file.split('.')[0] == "gdc_sample_sheet":
            df_patients = pd.read_csv(os.path.join(input_folder, file), sep="\t").set_index("File ID")
            break
    assert df_patients is not None, "'gdc_sample_sheet.tsv' is missing !"

    # Load files list
    rna_folder_path  = os.path.join(input_folder, 'RNA-seq')
    file_list = os.listdir(rna_folder_path)

    # Create output matrices
    for case_file in tqdm(df_patients.index, total=len(df_patients.index)):
        if case_file in file_list:
            case_path = os.path.join(rna_folder_path, case_file)
            file_path = None
            for f in os.listdir(case_path):
                if f.endswith(".tsv"):
                    file_path = os.path.join(case_path, f)
            if file_path is not None:
                df_rnaseq = pd.read_csv(file_path, sep="\t", skiprows=1).iloc[4:, :].set_index("gene_id")
                counts_tpm = pd.Series({"Case ID": df_patients.loc[case_file, "Case ID"] ,
                                        "Sample Type": df_patients.loc[case_file, "Sample Type"]})
#                 counts_fpkm = counts_tpm.copy()
#                 counts_fpkm_uq = counts_tpm.copy()

                counts_tpm = pd.concat([counts_tpm, df_rnaseq["tpm_unstranded"]])
#                 counts_fpkm = pd.concat([counts_fpkm, df_rnaseq["fpkm_unstranded"]])
#                 counts_fpkm_uq = pd.concat([counts_fpkm_uq, df_rnaseq["fpkm_uq_unstranded"]])

                gene_counts_tpm.append(counts_tpm.rename(df_patients.loc[case_file, "Sample ID"]))
#                 gene_counts_fpkm.append(counts_fpkm.rename(df_patients.loc[case_file, "Sample ID"]))
#                 gene_counts_fpkm_uq.append(counts_fpkm_uq.rename(df_patients.loc[case_file, "Sample ID"]))
            else:
                print(
                    "RNA-seq data of sample " + df_patients.loc[ind, "Sample ID"] + " are missing ! Check file " + ind)
        else:
            print("RNA-seq data of sample " + df_patients.loc[ind, "Sample ID"] + " are missing ! Check file " + ind)

    gene_counts_tpm = pd.concat(gene_counts_tpm, axis=1)
#     gene_counts_fpkm = pd.concat(gene_counts_fpkm, axis=1)
#     gene_counts_fpkm_uq = pd.concat(gene_counts_fpkm_uq, axis=1)

    # Create annotation matrix
    assert file_path is not None, "No rnaseq file was found, check your folder architecture !"
    gene_annotations = df_rnaseq[['gene_name', 'gene_type']]

    # Create output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'merged_RNAseq')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gene_counts_tpm.to_csv(os.path.join(output_folder, "tpm.csv"))
#     gene_counts_fpkm.to_csv(os.path.join(output_folder, "fpkm.csv"))
#     gene_counts_fpkm_uq.to_csv(os.path.join(output_folder, "fpkm_uq.csv"))
    gene_annotations.to_csv(os.path.join(output_folder, "gene_annotations.csv"))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input file path")
    parser.add_argument("-o", "--output", type=str, help="output path", default=None)
    args = parser.parse_args()
    main(input_folder=args.input, output_folder=args.output)