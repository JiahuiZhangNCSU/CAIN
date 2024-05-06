from CAIN.utils import AlphaCavity
import os 

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument('--job_name', type=str, default='test', help='Name of the job, needs to be the same as the processing folder name')
    parser.add_argument('--min_radius', type=float, default=3.0, help='Minimum radius of the alpha sphere cavity')
    parser.add_argument('--max_radius', type=float, default=6.0, help='Maximum radius of the alpha sphere cavity')
    parser.add_argument('--tunnel_d', type=float, default=2.8, help='Distance cutoff for the tunnel cavities')
    parser.add_argument('--aa', type=bool, default=False, help='Whether to use all atom representation')
    parser.add_argument('--save_pdb', type=bool, default=True, help='Whether to save the pdb files')
    parser.add_argument('--do_clustering', type=bool, default=False, help='Whether to do the pockets clustering')
    parser.add_argument('--save_csv', type=bool, default=True, help='Whether to save the csv files')
    parser.add_argument('--get_all_volumes', type=bool, default=False, help='Whether to get all the volumes of the cavities')
    parser.add_argument('--get_ligand_pocket_volume', type=bool, default=False, help='Whether to get the volume of the ligand pocket')
    parser.add_argument('--get_ligand_tunnel_length', type=bool, default=False, help='Whether to get the length of the tunnel')
    parser.add_argument('--tunnel_is_open', type=bool, default=False, help='Whether to consider the tunnel as an open tunnel')

    args = parser.parse_args()

    # save the parameters to a json file.
    params = vars(args)
    json_file = args.job_name + 'params.json'
    with open(json_file, 'w') as f:
        json.dump(params, f)

    ########## Change here for the naming input files ##########
    protein_pdb_file = args.job_name+'_protein_processed.pdb'
    if os.exists(args.job_name+'_ligand.pdb'):
        ligand_file = args.job_name+'_ligand.pdb'
    elif os.exists(args.job_name+'_ligand.sdf'):
        ligand_file = args.job_name+'_ligand.sdf'
        

    alpha_cavity = AlphaCavity(max_radius=args.max_radius, min_radius=args.min_radius, tunnel_d=args.tunnel_d, aa=args.aa, is_open= args.tunnel_is_open)
    # initialize the alpha spheres.
    alpha_cavity.get_AlphaSpheres(protein_pdb_file)
    # initialize the ligand crds.
    alpha_cavity.get_ligand_crds(ligand_file)
    # get the pockets.
    if args.do_clustering:
        alpha_cavity.get_pockets_clustering()
    else:
        alpha_cavity.get_pockets_as_whole()
    # get the ligand pocket.
    alpha_cavity.get_ligand_pocket()
    # get the ligand tunnel.
    alpha_cavity.get_ligand_tunnel()
    # write pdb.
    if args.save_pdb:
        alpha_cavity.save_pockets_pdb()
        alpha_cavity.save_ligand_pocket_pdb()
    # write csv.
    if args.save_csv:
        alpha_cavity.save_pockets_csv()
        alpha_cavity.save_ligand_pocket_csv()
    volumes, ligand_pocket_volume, ligand_tunnel_length = None, None, None
    # get all volumes.
    if args.get_all_volumes:
        volumes = alpha_cavity.get_all_volumes()
    # get the ligand pocket volume.
    if args.get_ligand_pocket_volume:
        ligand_pocket_volume = alpha_cavity.get_ligand_pocket_volume()
    # get the ligand tunnel length.
    if args.get_ligand_tunnel_length:
        ligand_tunnel_length = alpha_cavity.get_ligand_tunnel_length()
    # save the results to a json file.
    results = {'volumes': volumes, 'ligand_pocket_volume': ligand_pocket_volume, 'ligand_tunnel_length': ligand_tunnel_length}
    results_file = os.path.join(args.job_name, 'outputs.json')
    with open(results_file, 'w') as f:
        json.dump(results, f)
