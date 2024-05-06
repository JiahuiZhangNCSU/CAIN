from CAIN.IRWD import compare_two_pockets

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--P', type=str, help='The path to the first pocket file (the less alpha-spheres)')
    parser.add_argument('--Q', type=str, help='The path to the second pocket file (the more alpha-spheres)')
    parser.add_argument('--eps', type=float, default=1.0, help='The epsilon for the IRWD')
    parser.add_argument('--save_rotation', type=bool, default=False, help='Whether to save the rotation matrix to a file')

    args = parser.parse_args()

    P, Q = args.P, args.Q

    score, rotation = compare_two_pockets(P, Q, args.eps)

    with open('IRWD_output.txt', 'w') as f:
        f.write(f'The IRWD score is {score}\n')
        if args.save_rotation:
            f.write(f'The rotation matrix is \n{rotation.as_matrix()}')