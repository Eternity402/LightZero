from easydict import EasyDict
import pdb
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 200
update_per_collect = 100
batch_size = 256
max_env_step = int(1e7)
reanalyze_ratio = 0.
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
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 8, 8), # channel goes first / maybe just snapshot is okay, so not many channel needed?
            action_space_size=64,
            image_channel=3,
            # We use the small size model for tictactoe.
            num_res_blocks=4,
            num_channels=128,
            support_scale=600,
            reward_support_size=1201,
            value_support_size=1201, # let's use similar setting to connect4
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=int(60),
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.001,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=60,
        # num_unroll_steps=3,
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(5e3),
        replay_buffer_size=int(4e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
tictactoe_muzero_config = EasyDict(tictactoe_muzero_config)
main_config = tictactoe_muzero_config

tictactoe_muzero_create_config = dict(
    env=dict(
        type='othello',
        import_names=['zoo.board_games.othello.envs.othello_env'],
    ),
    env_manager=dict(type='subprocess'),
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
