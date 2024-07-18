import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "./data/", type = str)
    parser.add_argument('--dataset_name', default = 'FB15K237_sampled_txt', type = str, \
                        choices = [ 'FB15K237_sampled_txt'])
    parser.add_argument('--decoder', default = 'Semantic_Matching', type = str, \
                        choices = ['Semantic_Matching', 'Translational_Distance'])
    parser.add_argument('--exp', default = 'kgbound', type = str)
    parser.add_argument('-m', '--margin', default = 0.5, type = float)
    parser.add_argument('-lr', '--learning_rate', default = 5e-5, type = float)
    parser.add_argument('-L', '--num_RAMPLayer', default = 2, type = int)
    parser.add_argument('-d', '--dimension', default = 96, type = int)
    parser.add_argument('-phi', '--phi', default = 'LeakyReLU', type = str, choices = ['LeakyReLU', 'Identity'])
    parser.add_argument('-rho', '--rho', default = 'Identity', type = str, choices = ['LeakyReLU', 'Identity'])
    parser.add_argument('-psi', '--psi', default = 'Identity', type = str, choices = ['LeakyReLU', 'Identity'])
    parser.add_argument('--aggr', default = "mean", type = str, choices = ['mean', 'sum'])
    parser.add_argument('-s', '--norm', default = 15.0, type = float)
    parser.add_argument('--seed', default = 0, type = int)

    parser.add_argument('-e', '--num_epoch', default = 2000, type = int)
    parser.add_argument('-b', '--num_batch', default = 1, type = int)

    args = parser.parse_args()

    return args