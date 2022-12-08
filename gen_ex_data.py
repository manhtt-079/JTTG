import argparse
# from config.config import gen_data_preparation_conf
from config.dataset import Reddit_TIFU_DataPreparationConf
from modules.datasets.process_data import RedditTIFUDataPreparation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/dataset.ini', help="Path to the config file")
    parser.add_argument('--name', type=str, default='reddit_tifu', help="Path to the config file")
    parser.add_argument('--vi', type=bool, default=False)

    args = parser.parse_args()
    config = Reddit_TIFU_DataPreparationConf(config_file=args.config)
    if args.name == 'reddit_tifu':
        processor = RedditTIFUDataPreparation(conf=config)
    
    processor.build_data()
    # conf = gen_data_preparation_conf(config_file=args.config, name=args.name)
    # if args.name in ['reddit_tifu', 'bill_sum']:
    #     processor = ENDataPreparation(conf=conf)
    # else:
    #     processor = VIDataPreparation(conf=conf)    
    # processor.process_and_save_data()
    
    
if __name__=='__main__':
    main()