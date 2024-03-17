from nni.experiment import Experiment


experiment = Experiment('local')
experiment.config.trial_command = 'python nn_choose_params.py'
experiment.config.trial_code_directory = '.'  # 代码目录
search_space = {
    'features1': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'features2': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'batch_size': {'_type': 'choice', '_value': [32, 64, 128, 256]}
}
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
experiment.run(8080)

# 防止自动退出
input('Press Enter to exit...')