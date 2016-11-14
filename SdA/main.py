import sys
import numpy as np

from experiment import Experiment

if __name__ == '__main__':
    np.set_printoptions(suppress=False)

    if (len(sys.argv) < 2):
        print "\nThe arguments must be: config_parameter_file\n"
        sys.exit(1);

    experiment = Experiment(sys.argv[1])
