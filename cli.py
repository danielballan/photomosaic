import argparse

def dimension(s):
    try:
        if 'x' in s:
            return tuple(map(int, s.split('x')))
        else:
            return int(s)    
    except:
        raise argparse.ArgumentTypeError('Dimensions must be either a single integer or of the form 5x4')

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('database')
    parser.add_argument('-tune', action='store_true')
    parser.add_argument('-d', '--dimensions', default='10x10', type=dimension)
    parser.add_argument('-r', '--recursion_level', default=0, type=int)
    return parser
