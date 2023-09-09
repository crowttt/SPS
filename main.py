import argparse
from process import Process
from pair2score.pair2score import convert
from db.session import session
from eval.eval import evaluation
from utils import load_arg


def main():
    parser = argparse.ArgumentParser(description='Processor collection')

    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default='config/kinetics-skeleton/train.yaml', help='path to the configuration file')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

    arg = parser.parse_args()

    process = Process(arg)
    process.start()

    conv = convert(arg)
    conv.convert()

    eva = evaluation(load_arg(arg.config))
    eva.train()
    eva.test()
    session.close()


if __name__ == '__main__':
    main()
