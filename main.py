import argparse
from process import Process
# from baseline.random_process import RandomProcess
from db.session import session
from eval.eval import evaluation
from utils import load_arg


def main():
    parser = argparse.ArgumentParser(description='Processor collection')

    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default='config/kinetics-skeleton/train.yaml', help='path to the configuration file')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

    arg = parser.parse_args()
    # rand_process = RandomProcess(arg)
    # rand_process.start()
    #----------------------
    process = Process(arg)
    process.start()
    # trainer = evaluation(load_arg(arg.config))
    # trainer.train()
    # trainer.test()
    session.close()


if __name__ == '__main__':
    main()
