import configparser

class DataPreparationConf:
    def __init__(self, config_file: str):
        self.conf = self.read_conf(conf_file=config_file)
        
        self.random_state = int(self.conf['base_data']['random_state'])
        self.n_processes = int(self.conf['base_data']['n_processes'])
        self.src_col_name = self.conf['base_data']['src_col_name']
        self.tgt_col_name = self.conf['base_data']['tgt_col_name']
        self.min_nsents = int(self.conf['base_data']['min_nsents'])
        self.max_nsents = int(self.conf['base_data']['max_nsents'])
        self.min_ntokens = int(self.conf['base_data']['min_ntokens'])
        self.max_ntokens = int(self.conf['base_data']['max_ntokens'])
        self.top_k = int(self.conf['base_data']['top_k'])

        
    def __repr__(self) -> str:
        return str(self.__dict__)
    
    
    @staticmethod
    def read_conf(conf_file) -> configparser.ConfigParser:
        config =  configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(conf_file)
    
        return config
            
class Reddit_TIFU_DataPreparationConf(DataPreparationConf):
    def __init__(self, config_file: str):
        super().__init__(config_file)
        
        self.sec_name = 'reddit_tifu'
        self.file_path = self.conf[self.sec_name]['file_path']
        self.long_dir = self.conf[self.sec_name]['long_dir']
        self.short_dir = self.conf[self.sec_name]['short_dir']
        self.min_nsents = int(self.conf[self.sec_name]['min_nsents'])
        self.max_nsents = int(self.conf[self.sec_name]['max_nsents'])
        
        self.test_size_long = int(self.conf[self.sec_name]['test_size_long'])
        self.test_size_short = int(self.conf[self.sec_name]['test_size_short'])
        
    
class BillSum_DataPreparationConf(DataPreparationConf):
    def __init__(self, config_file: str):
        super().__init__(config_file)
        
        self.sec_name = 'bill_sum'
        self.us_train_path = self.conf[self.sec_name]['us_train_path']
        self.us_test_path = self.conf[self.sec_name]['us_test_path']
        self.ca_test_path = self.conf[self.sec_name]['ca_test_path']
        self.test_size = int(self.conf[self.sec_name]['test_size'])
        self.output_dir = self.conf[self.sec_name]['output_dir']
    
class VnDS_DataPreparationConf(DataPreparationConf):
    def __init__(self, config_file: str):
        super().__init__(config_file)
        
        self.sec_name = 'vnds'
        self.data_path = self.conf[self.sec_name]['data_path']
        self.output_dir = self.conf[self.sec_name]['output_dir']
        self.max_nsents = int(self.conf[self.sec_name]['max_nsents'])


if __name__ == '__main__':
    pass