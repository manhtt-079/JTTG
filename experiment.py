EXPERIMENT_MAP = {
    'task1': {
        'dataset_name': 'reddit_tifu',
        'model_name': 'bart-sum',
        'is_long': True
    },
    'task2': {
        'dataset_name': 'bill_sum',
        'model_name': 'bart-sum',
        'use_us_test': True
    },
    'task3': {
        'dataset_name': 'reddit_tifu',
        'model_name': 't5-sum',
        'is_long': True
    },
    'task4': {
        'dataset_name': 'bill_sum',
        'model_name': 't5-sum',
        'use_us_test': True
    },
    'task5': {
        'dataset_name': 'vnds',
        'model_name': 'bartpho-sum'
    },
    'task6': {
        'dataset_name': 'vnds',
        'model_name': 'vit5-sum'
    },
    'task7': {
        'dataset_name': 'reddit_tifu',
        'model_name': 'pegasus-sum'
    }
}

if __name__=='__main__':
    pass