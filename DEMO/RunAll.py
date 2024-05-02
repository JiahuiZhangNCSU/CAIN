# run the analysis for all the folders in current directory.
import os
import subprocess

def single_run(folder):
    job_name = folder
    ########## Change here for the parameters resetting ##########
    command = f"python AlphaCacity.py --job_name {job_name} --get_ligand_pocket_volume True --get_ligand_tunnel_length True"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    folders = os.listdir()
    print("CAIN begins to run the analysis for all the folders in current directory.")
    for folder in folders:
        os.chdir(folder)
        try:
            single_run(folder)
        except:
            print(f"Error: CAIN cannot analyze {folder}. Analyzing the next folder.")
        os.chdir("..")
    print("All the analysis have been done.")
    
