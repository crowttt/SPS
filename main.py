import argparse


def main():
    parser = argparse.ArgumentParser(description='Processor collection')

    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

    

if __name__ == '__main__':
    main()