import os
import sys
from pathlib import Path
from GPUtil import showUtilization as gpu_usage
from tensorboardX import SummaryWriter


base_dir = Path(__file__).absolute().parent
sys.path.append(os.path.join(base_dir, 'monodepth2'))
sys.path.append(os.path.join(base_dir, 'manydepth'))

from collections import deque
import numpy as np
import torch
from src.agents import RandomAgent, HeuristicRandomAgent, ChamferAgent, DepthMAEAgent, MyAgent, FrontAgent, ForwardAgent
from src.config import get_config

from src.env_wrapper import SimEnv
from src.my_utils import MapStat, Timings 
from src.options import Options

def find_nearest(num, num_list):
    num_array = np.array(num_list)
    closest_num = num_array[np.abs(num_array - num).argmin()]
    return closest_num


def main():
    options = Options()
    opt = options.parse()

    if opt.model_type == 'flow2depth':
        from src.models import MyTrainer as ModelTrainer
    else:
        raise NotImplementedError('Model {} is not implemented'.format(opt.model_type))

    model_trainer = ModelTrainer(opt)
    cfg = get_config(opt.cfg_path)

    model_type = 'manydepth' if opt.model_type in ['manydepth', 'flow2depth'] else 'monodepth'

    env = SimEnv(cfg, opt.scene_id)
    if opt.agent_type == 'random':
        agent = RandomAgent(env, cfg.AGENT)
    elif opt.agent_type == 'forward':
        agent = ForwardAgent(env, cfg.AGENT)
    elif opt.agent_type == 'frontier':
        agent = FrontAgent(env, cfg.AGENT, model_type, opt.video_interval)
    elif opt.agent_type == 'mix':
        agent = MyAgent(env, cfg.AGENT, model_type, opt.video_interval)
    
    if opt.agent_type not in ['random', 'forward']:
        dummy = HeuristicRandomAgent(env, cfg.AGENT)
    else: 
        dummy = agent

    if type(agent) in [FrontAgent, MyAgent]:
        print('With frontier explor!')  
    else:
        print('No explor' )

    logger = SummaryWriter(model_trainer.log_path)
    map_logger = MapStat(env)
    timer = Timings()
    timer.reset()

    ## Init test as the starting of curve
    data_count = 0
    APPROX_RANGE = [i for i in range(cfg.TRAIN.WARMUP_DATA, cfg.TRAIN.FULL_DATA_LEN, cfg.COLLECT_STEPS)]
    print(APPROX_RANGE)
    metrics = model_trainer.test(logger, model_trainer.step)
    acc = metrics['test/a1']()
    logger.add_scalar('Explor/a1_wrt_data_num', acc,  data_count)
    ##

    count_steps = 0
    action = ['move_forward'] * 20
    observations, _ = env.step(action)  # noqa: F841

    training_obs = deque(maxlen=cfg.TRAIN.Q_MAXLEN)

    while len(training_obs) < cfg.TRAIN.WARMUP_DATA:
        training_obs += observations
        observations = dummy.act(model_trainer.model, observations)

    data_count = len(training_obs)

    logger.add_scalar('Explor/data_num', len(list(training_obs)), model_trainer.step,)

    # Old version is True False False
    _ = model_trainer.train_one_step(list(training_obs), logger, count_steps)
    torch.cuda.empty_cache()

    reset_done_flag = False
    #while count_steps <= cfg.COLLECT_NUM:
    while len(training_obs) < cfg.TRAIN.FULL_DATA_LEN: 
        if len(training_obs) > cfg.TRAIN.Q_MAXLEN / 5:

            if opt.reset_halfway and len(training_obs) > 18000 and (not reset_done_flag):
                loop_num = 3
                reset_done_flag = True
                model_trainer.reset_model()
            else:
                loop_num = 2

            for _ in range(loop_num):
                acc = model_trainer.train_one_step(list(training_obs), logger, count_steps)
                torch.cuda.empty_cache()
            model_trainer.log_trajectory(env, list(training_obs), logger, count_steps)

            log_data_approx = find_nearest(data_count, APPROX_RANGE)
            logger.add_scalar('Explor/data_num', log_data_approx, model_trainer.step,)
            logger.add_scalar('Explor/a1_wrt_data_num', acc,  log_data_approx)
            print('Training one iter, data count: {}'.format(data_count))
        timer.time("Model train")

        obs_len = 0
        print('Collect data')
        while obs_len < cfg.COLLECT_STEPS:
            observations = agent.act(model_trainer.model, list(training_obs))
            obs_len += len(observations)
            training_obs += observations
            data_count += len(observations)

        map_logger.step([d['pos'] for d in list(training_obs)])
        timer.time("Env step")


        tgt_index = opt.frame_ids[1]
        if (count_steps % (opt.log_frequency // 20) == 0):
            buf, percent = map_logger.summary()
            logger.add_image('Explor/map', buf, count_steps)
            logger.add_scalar('Explor/percent', percent, count_steps)
            if type(agent) in [FrontAgent, MyAgent]:
                buf = agent.map.visualize(None,None, 'buf')
                logger.add_image('Explor/costmap', buf, count_steps)
                if hasattr(agent, 'cur_costmap') and (agent.cur_costmap is not None):
                    buf = agent.cur_costmap.visualize(None, None, 'buf')
                    logger.add_image('Explor/insist_cur', buf, count_steps)
                    buf = agent.past_costmap.visualize(None, None, 'buf')
                    logger.add_image('Explor/insist_pst', buf, count_steps)
                agent.map.reset_viz()
            timer.time("Model eval")

            print('Steps = {}'.format(count_steps), timer.summary())
        count_steps += 1

    # Collect data end, 
    buf, percent = map_logger.summary()
    logger.add_image('Explor/map', buf, count_steps+1)
    logger.add_scalar('Explor/percent', percent, count_steps+1)
    if type(agent) in [FrontAgent, MyAgent]:
        buf = agent.map.visualize(None,None, 'buf')
        logger.add_image('Explor/costmap', buf, count_steps+1)
        agent.map.reset_viz()

    for i in range(4): 
        _ = model_trainer.train_one_step(list(training_obs), logger, count_steps)
        model_trainer.test(logger, model_trainer.step)

    model_trainer.test(logger, model_trainer.step, final=True)

    del training_obs
    del model_trainer
    del env

if __name__ == '__main__':
    main()
        


