import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run SimGNN.")

    parser.add_argument("--cuda",
                        dest="cuda",
                        action="store_true",
                        help="Run the code with cuda.")

    parser.add_argument("--test",
                        dest="test",
                        action="store_true",
                        help="Run testing script.")

    parser.add_argument("--val",
                        dest="val",
                        action="store_true",
                        help="Run validation script.")
                    
    parser.add_argument("--dataset",
                        nargs = "?",
                        default = "AIDS700nef",
	                help = "Dataset name. Default is AIDS700nef")

    parser.add_argument("--gnn-operator",
                        nargs = "?",
                        default = "gcn",
	                help = "Type of GNN-Operator. Default is gcn")

    parser.add_argument("--epochs",
                        type = int,
                        default = 350,
	                help = "Number of training epochs. Default is 350.")

    parser.add_argument("--filters-1",
                        type = int,
                        default = 64,
	                help = "Filters (neurons) in 1st convolution. Default is 64.")

    parser.add_argument("--filters-2",
                        type = int,
                        default = 32,
	                help = "Filters (neurons) in 2nd convolution. Default is 32.")

    parser.add_argument("--filters-3",
                        type = int,
                        default = 16,
	                help = "Filters (neurons) in 3rd convolution. Default is 16.")

    parser.add_argument("--tensor-neurons",
                        type = int,
                        default = 16,
	                help = "Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type = int,
                        default = 16,
	                help = "Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--batch-size",
                        type = int,
                        default = 128,
	                help = "Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--bins",
                        type = int,
                        default = 16,
	                help = "Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0,
	                help = "Dropout probability. Default is 0.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.001,
	                help = "Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type = float,
                        default = 5*10**-4,
	                help = "Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--enable-astar",
                        action="store_true",
                        help="Enable a-star algorithm for GED.")

    parser.add_argument("--astar-beamwidth",
                        type=int,
                        default=0,
                        help="Size of beam-search witdh (0 = no beam). Default is 0.")

    parser.add_argument("--astar-trustfact",
                        type=float,
                        default=1.,
                        help="The trust factor on GNN prediction (0 = no GNN). Default is 1.")

    parser.add_argument("--astar-nopred",
                        type=float,
                        default=0.,
                        help="If the smaller graph has no more than x nodes, stop using heuristics.")

    parser.add_argument("--astar-use-net",
                        action="store_true",
                        help="Set this parametr to enable neural net prediciton in A-star (by default use hungarian heuristic)")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")
                        
    parser.add_argument("--diffpool",
                        dest="diffpool",
                        action="store_true",
	                help = "Enable differentiable pooling.")

    parser.add_argument("--plot",
                        dest="plot",
                        action="store_true")
                        
    parser.add_argument("--synth",
                        dest="synth",
                        action="store_true")
                        
    parser.add_argument("--measure-time",
                        action="store_true",
                    help = "Measure average calculation time for one graph pair. Learning is not performed.")
    
    parser.add_argument("--notify",
                        dest="notify",
                        action="store_true",
                    help = "Send notification message when the code is finished (only Linux & Mac OS support).")

    parser.set_defaults(histogram=False)
    parser.set_defaults(diffpool=False)
    parser.set_defaults(plot=False)
    parser.set_defaults(measure_time=False)
    parser.set_defaults(notify=False)
    parser.set_defaults(synth=False)

    return parser.parse_args()
