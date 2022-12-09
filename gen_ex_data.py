import argparse
# from config.config import gen_data_preparation_conf
from config.dataset import Reddit_TIFU_DataPreparationConf, BillSum_DataPreparationConf, VnDS_DataPreparationConf
from modules.datasets.process_data import RedditTIFUDataPreparation, BillSumDataPreparation, VNDSDataPreparation


PROCESSOR_ARCHIVE_MAP = {
    'reddit_tifu': {'conf': Reddit_TIFU_DataPreparationConf, 'processor': RedditTIFUDataPreparation},
    'bill_sum': {'conf': BillSum_DataPreparationConf, 'processor': BillSumDataPreparation},
    'vnds': {'conf': VnDS_DataPreparationConf, 'processor': VNDSDataPreparation}
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/dataset.ini', help="Path to the config file")
    parser.add_argument('--name', type=str, default='reddit_tifu', help="Path to the config file")
    parser.add_argument('--vi', type=bool, default=False)

    args = parser.parse_args()
    if args.name not in PROCESSOR_ARCHIVE_MAP:
        raise ValueError(f"Dataset name must be in: {PROCESSOR_ARCHIVE_MAP.keys()}")
    pair = PROCESSOR_ARCHIVE_MAP[args.name]
    config = pair['conf'](config_file=args.conf)
    processor = pair['processor'](conf=config)
    
    processor.build_data()
    
if __name__=='__main__':
    main()