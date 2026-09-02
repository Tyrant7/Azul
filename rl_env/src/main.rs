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
        "new_baseline_invariants",
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
        writer.flush();

        println!(
            "iteration={} timesteps={} actor_loss={:.4} critic_loss={:.4} actor_grad={:.3} critic_grad={:.3} kl={:.4} clip={:.3} score_diff={:.2} winner_score: {:.2} win_rate={:.3}",
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
        );
    });
    Ok(())
}
