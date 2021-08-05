from src.utils import tab_printer
from src.parser import parameter_parser
from src.astar_genn import GENNTrainer as Trainer
import torch

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a genn model.
    """
    args = parameter_parser()
    tab_printer(args)

    trainer = Trainer(args)

    if args.cuda:
        trainer.model = trainer.model.cuda()

    if args.test or args.val:
        if args.enable_astar:
            if args.astar_use_net:
                weight_path = 'best_genn_{}_{}_astar.pt'
                trainer.model.load_state_dict(torch.load(weight_path.format(args.dataset, args.gnn_operator)))
        else:
            weight_path = 'best_genn_{}_{}.pt'
            trainer.model.load_state_dict(torch.load(weight_path.format(args.dataset, args.gnn_operator)))
        trainer.model.eval()
        trainer.score(test=args.test)
        exit(0)
    else: # training
        if args.enable_astar:
            trainer.model.load_state_dict(torch.load('best_genn_{}_{}.pt'.format(args.dataset, args.gnn_operator)))
        trainer.fit()
        weight_path = 'best_genn_{}_{}_astar.pt' if args.enable_astar else 'best_genn_{}_{}.pt'
        trainer.model.load_state_dict(torch.load(weight_path.format(args.dataset, args.gnn_operator)))
        trainer.model.eval()
        if not args.enable_astar:
            trainer.score()
    
if __name__ == "__main__":
    main()
