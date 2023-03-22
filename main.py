import argparse
from process import Process
from db.session import session


def main():
    parser = argparse.ArgumentParser(description='Processor collection')

    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default='config/kinetics-skeleton/train.yaml', help='path to the configuration file')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

    arg = parser.parse_args()
    process = Process(arg)
    process.start()
    session.close()


if __name__ == '__main__':
    main()
