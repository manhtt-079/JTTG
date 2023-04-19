import argparse

from config.dataset import (
    Reddit_TIFU_DataPreparationConf,
    BillSum_DataPreparationConf,
    VnDS_DataPreparationConf,
    DataPreparationConf,
    ViNewsQA_DataPRConf,
    ViQuAD_DataPRConf,
    GovReportDataPRConf
)

from modules.datasets.process_data import (
    RedditTIFUDataPreparation,
    BillSumDataPreparation,
    VNDSDataPreparation,
    ViNewsQADataPreparation,
    ViQuADDataPreparation,
    GovReportDataPreparation
)


PROCESSOR_ARCHIVE_MAP = {
    'reddit_tifu': {'conf': Reddit_TIFU_DataPreparationConf, 'processor': RedditTIFUDataPreparation},
    'bill_sum': {'conf': BillSum_DataPreparationConf, 'processor': BillSumDataPreparation},
    'vnds': {'conf': VnDS_DataPreparationConf, 'processor': VNDSDataPreparation},
    'vinewsqa': {'conf': ViNewsQA_DataPRConf, 'processor': ViNewsQADataPreparation},
    'viquad': {'conf': ViQuAD_DataPRConf, 'processor': ViQuADDataPreparation},
    'gov-report': {'conf': GovReportDataPRConf, 'processor': GovReportDataPreparation}
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config/dataset.ini', help="Path to the config file")
    parser.add_argument('--dataset', type=str, default='reddit_tifu', help="Path to the config file")

    args = parser.parse_args()
    if args.dataset not in PROCESSOR_ARCHIVE_MAP:
        raise ValueError(f"Dataset: {args.dataset} must be in: {PROCESSOR_ARCHIVE_MAP.keys()}")
    
    processor_map = PROCESSOR_ARCHIVE_MAP[args.dataset]
    config = processor_map['conf'](config_file=args.config_file)
    processor: DataPreparationConf = processor_map['processor'](conf=config)
    
    processor.build_data()
    
if __name__=='__main__':
    main()