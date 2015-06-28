import argparse

def dimension(s):
    try:
        if 'x' in s:
            return tuple(map(int, s.split('x')))
        else:
            return int(s)    
    except:
        raise argparse.ArgumentTypeError('Dimensions must be either a single integer or of the form 5x4')

def args_parser(output_file=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('database')
    if output_file:
        parser.add_argument('outfile')
    parser.add_argument('-tune', action='store_true')
    parser.add_argument('-d', '--dimensions', default='10x10', type=dimension)
    parser.add_argument('-r', '--recursion_level', default=0, type=int)
    parser.add_argument('-f', '--folders', nargs='*', default=[])
    parser.add_argument('-m', '--mask')
    return parser
    
def get_database(args):
    from sql_image_pool import SqlImagePool
    pool = SqlImagePool(args.database)
    for folder in args.folders:
        pool.add_directory(folder)
    return pool    


if __name__=='__main__':
    from photomosaic import Photomosaic
    
    args = args_parser(True).parse_args()

    pool = get_database(args)

    p = Photomosaic(args.infile, pool, tuning=args.tune, mask=args.mask)
    p.partition_tiles(args.dimensions, depth=args.recursion_level)
    p.match()
    p.assemble()
    p.save(args.outfile)
    
    pool.close()
