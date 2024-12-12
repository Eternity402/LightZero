from easydict import EasyDict
import os
import pdb
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 50 # 200
update_per_collect = 50 # 이거 오히려 더 줄어야 할듯. 샘플이 너무 적어서 학습하기에 충분치 않아보임 - 아니라네.. 흠 뭐 어차피 루트야 루트고 학습은 별개니까
batch_size = 512
max_env_step = int(1e10)
reanalyze_ratio = 0.

os.environ["checkpoint_iter"] = "0"
os.environ["need_check"] = "False"
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

tictactoe_muzero_config = dict(
    exp_name=f'data_muzero/othello_muzero_sp-mode_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        # prob_random_agent=0.3,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        save_replay_gif=True,
        replay_path_gif='./replay_gif',
        agent_vs_human=True,
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 8, 8), # channel goes first / maybe just snapshot is okay, so not many channel needed?
            action_space_size=65,
            image_channel=3,
            # We use the small size model for tictactoe.
            num_res_blocks=3, # 4
            num_channels=128, # 128
            support_scale=32,
            reward_support_size=65,
            value_support_size=65, # let's use similar setting to connect4
            self_supervised_learning_loss=True,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        # model_path=None,
        model_path='/Users/gimgeonsu/Study/LightZero/data_muzero/othello_muzero_sp-mode_ns50_upc50_rer0.0_seed0_241204_205115/ckpt/iteration_270000.pth.tar',
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=int(80),
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.001,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=10,
        num_unroll_steps=5,
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=0.998,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(3e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        ssl_loss_weight=2
    ),
)
tictactoe_muzero_config = EasyDict(tictactoe_muzero_config)
main_config = tictactoe_muzero_config

tictactoe_muzero_create_config = dict(
    env=dict(
        type='othello',
        import_names=['zoo.board_games.othello.envs.othello_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
tictactoe_muzero_create_config = EasyDict(tictactoe_muzero_create_config)
create_config = tictactoe_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
