
from train import *

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description='Script to test DKT.')
    parser.add_argument('--dataset', type=str, default='junyi_test', help='')
    parser.add_argument('--hidden_num', type=int, default=512, help='')

    parser.add_argument('--seed', type=int, default=0, help='')

    parser.add_argument('--cv_num', type=int, default=1, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--q_num', type=int, default=720, help='')#######
    parser.add_argument('--length', type=int, default=300, help='')#######
    parser.add_argument('--time_spend', type=int, default=172, help='')#######
    parser.add_argument('--d_model', type=int, default=512, help='')#######
    parser.add_argument('--nhead', type=int, default=8, help='')#######
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='')#######
    parser.add_argument('--dropout', type=float, default=0, help='')#######
    parser.add_argument('--gpu', type=str, default='1', help='')#######
    parser.add_argument('--speed_cate', type=int, default='10', help='')#######
    parser.add_argument('--loss_rate', type=float, default='0.25', help='')#######

    params = parser.parse_args()
    dataset = params.dataset
    if dataset == 'junyi_text':
        params.q_num = 285
        params.length = 200
    if dataset == 'junyi_test':
        params.q_num = 720
        params.length = 200
    if dataset == 'junyi_all':
        params.q_num = 720
        params.length = 200
        

    experiment(
        dataset = params.dataset,
        hidden_num = params.hidden_num,
        learning_rate = params.learning_rate,
        epochs = params.epochs,
        batch_size = params.batch_size,
        seed = params.seed,
        cv_num=params.cv_num,
    
        q_num=params.q_num ,
        length=params.length,
        time_spend=params.time_spend,
        d_model=params.d_model,
        nhead=params.nhead,
        num_encoder_layers=params.num_encoder_layers,
        dropout=params.dropout,
        gpu=params.gpu,
        speed_cate=params.speed_cate,
        loss_rate=params.loss_rate

    )

