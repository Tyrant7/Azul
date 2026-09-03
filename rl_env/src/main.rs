use std::time::{SystemTime, UNIX_EPOCH};

use rl_env::get_device;
use tensorboard_rs::summary_writer::SummaryWriter;

fn main() -> Result<(), tch::TchError> {
    let config = rl_env::PpoConfig {
        timesteps_per_batch: 1_000,
        max_timesteps_per_episode: 1_000,
        updates_per_iteration: 5,
        ..Default::default()
    };

    let mut trainer = rl_env::PpoTrainer::new(config)?;
    let mut environment = rl_env::AzulEnv::new(0, config.max_timesteps_per_episode);
    let mut writer = SummaryWriter::new(format!(
        "runs/azul_ppo/{}-{}",
        "camf_split_LR",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs()
    ));

    println!("tch CUDA available: {}", tch::Cuda::is_available());
    println!("tch CUDA devices: {}", tch::Cuda::device_count());
    println!("training on device: {:?}", get_device());
    trainer.train_with_callback(&mut environment, 1_000_000, |metrics| {
        writer.add_scalar("loss/actor", metrics.actor_loss, metrics.timesteps);
        writer.add_scalar("loss/critic", metrics.critic_loss, metrics.timesteps);
        writer.add_scalar(
            "gradient/actor_norm",
            metrics.actor_grad_norm,
            metrics.timesteps,
        );
        writer.add_scalar(
            "gradient/critic_norm",
            metrics.critic_grad_norm,
            metrics.timesteps,
        );
        writer.add_scalar(
            "gradient/actor_clip_coefficient",
            metrics.actor_grad_clip_coefficient,
            metrics.timesteps,
        );
        writer.add_scalar(
            "gradient/critic_clip_coefficient",
            metrics.critic_grad_clip_coefficient,
            metrics.timesteps,
        );
        writer.add_scalar("policy/approx_kl", metrics.approx_kl, metrics.timesteps);
        writer.add_scalar(
            "policy/clip_fraction",
            metrics.clip_fraction,
            metrics.timesteps,
        );
        writer.add_scalar(
            "episode/mean_return",
            metrics.mean_episode_return,
            metrics.timesteps,
        );
        writer.add_scalar(
            "episode/mean_length",
            metrics.mean_episode_length,
            metrics.timesteps,
        );
        writer.add_scalar(
            "episode/mean_final_score_difference",
            metrics.mean_final_score_difference,
            metrics.timesteps,
        );
        writer.add_scalar(
            "episode/mean_winner_score",
            metrics.mean_winner_score,
            metrics.timesteps,
        );
        writer.add_scalar(
            "episode/player_zero_win_rate",
            metrics.player_zero_win_rate,
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_zero_average_penalties",
            metrics.average_penalties_per_game[0],
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_one_average_penalties",
            metrics.average_penalties_per_game[1],
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_zero_average_bonus_points",
            metrics.average_bonus_points_per_game[0],
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_one_average_bonus_points",
            metrics.average_bonus_points_per_game[1],
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_zero_average_rows_filled",
            metrics.average_rows_filled_per_game[0],
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_one_average_rows_filled",
            metrics.average_rows_filled_per_game[1],
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_zero_average_columns_filled",
            metrics.average_columns_filled_per_game[0],
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_one_average_columns_filled",
            metrics.average_columns_filled_per_game[1],
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_zero_average_tile_bonuses",
            metrics.average_tile_bonuses_per_game[0],
            metrics.timesteps,
        );
        writer.add_scalar(
            "game/player_one_average_tile_bonuses",
            metrics.average_tile_bonuses_per_game[1],
            metrics.timesteps,
        );
        writer.flush();

        println!(
            "iteration={} timesteps={} actor_loss={:.4} critic_loss={:.4} actor_grad={:.3} critic_grad={:.3} kl={:.4} clip={:.3} score_diff={:.2} winner_score: {:.2} win_rate={:.3} p0_penalties/game={:.2} p1_penalties/game={:.2} p0_bonuses/game={:.2} p1_bonuses/game={:.2} p0_rows/game={:.2} p1_rows/game={:.2} p0_columns/game={:.2} p1_columns/game={:.2} p0_tile_bonuses/game={:.2} p1_tile_bonuses/game={:.2}",
            metrics.iteration,
            metrics.timesteps,
            metrics.actor_loss,
            metrics.critic_loss,
            metrics.actor_grad_norm,
            metrics.critic_grad_norm,
            metrics.approx_kl,
            metrics.clip_fraction,
            metrics.mean_final_score_difference,
            metrics.mean_winner_score,
            metrics.player_zero_win_rate,
            metrics.average_penalties_per_game[0],
            metrics.average_penalties_per_game[1],
            metrics.average_bonus_points_per_game[0],
            metrics.average_bonus_points_per_game[1],
            metrics.average_rows_filled_per_game[0],
            metrics.average_rows_filled_per_game[1],
            metrics.average_columns_filled_per_game[0],
            metrics.average_columns_filled_per_game[1],
            metrics.average_tile_bonuses_per_game[0],
            metrics.average_tile_bonuses_per_game[1],
        );
    });
    Ok(())
}
