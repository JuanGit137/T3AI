from experiments import run_all_experiments
from dimreduct import reduce_dimensions
from plots import generate_plots
import warnings

warnings.filterwarnings("ignore")

def main():
    reduce_dimensions()
    run_all_experiments()
    generate_plots()

if __name__ == '__main__':
    main()