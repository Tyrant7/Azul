use std::time::{SystemTime, UNIX_EPOCH};

use rl_env::{PpoMetrics, get_device};
use tensorboard_rs::summary_writer::SummaryWriter;

pub struct TrainingLogger {
    writer: SummaryWriter,
}

impl TrainingLogger {
    pub fn new(run_name: &str) -> Self {
        let run_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        Self {
            writer: SummaryWriter::new(format!("runs/azul_ppo/{run_name}-{run_id}")),
        }
    }

    pub fn log_device(&self) {
        println!("tch CUDA available: {}", tch::Cuda::is_available());
        println!("tch CUDA devices: {}", tch::Cuda::device_count());
        println!("training on device: {:?}", get_device());
    }

    pub fn log(&mut self, metrics: &PpoMetrics) {
        let step = metrics.timesteps;
        self.writer
            .add_scalar("loss/actor", metrics.actor_loss, step);
        self.writer
            .add_scalar("loss/critic", metrics.critic_loss, step);
        self.writer
            .add_scalar("gradient/actor_norm", metrics.actor_grad_norm, step);
        self.writer
            .add_scalar("gradient/critic_norm", metrics.critic_grad_norm, step);
        self.writer.add_scalar(
            "gradient/actor_clip_coefficient",
            metrics.actor_grad_clip_coefficient,
            step,
        );
        self.writer.add_scalar(
            "gradient/critic_clip_coefficient",
            metrics.critic_grad_clip_coefficient,
            step,
        );
        self.writer
            .add_scalar("update/actor_norm", metrics.actor_update_norm, step);
        self.writer
            .add_scalar("update/critic_norm", metrics.critic_update_norm, step);
        self.writer
            .add_scalar("policy/approx_kl", metrics.approx_kl, step);
        self.writer
            .add_scalar("policy/clip_fraction", metrics.clip_fraction, step);

        for (name, value) in [
            ("return_mean", metrics.return_mean),
            ("return_std", metrics.return_std),
            ("return_min", metrics.return_min),
            ("return_max", metrics.return_max),
            ("value_mean", metrics.value_mean),
            ("value_std", metrics.value_std),
            ("value_min", metrics.value_min),
            ("value_max", metrics.value_max),
            ("advantage_mean", metrics.advantage_mean),
            ("advantage_std", metrics.advantage_std),
            ("advantage_min", metrics.advantage_min),
            ("advantage_max", metrics.advantage_max),
            ("explained_variance", metrics.explained_variance),
        ] {
            let tag = format!("critic/{name}");
            self.writer.add_scalar(&tag, value, step);
        }

        self.writer
            .add_scalar("episode/mean_return", metrics.mean_episode_return, step);
        self.writer
            .add_scalar("episode/mean_length", metrics.mean_episode_length, step);
        self.writer.add_scalar(
            "episode/mean_final_score_difference",
            metrics.mean_final_score_difference,
            step,
        );
        self.writer
            .add_scalar("episode/mean_winner_score", metrics.mean_winner_score, step);
        self.writer.add_scalar(
            "episode/player_zero_win_rate",
            metrics.player_zero_win_rate,
            step,
        );

        for (player, values) in [("player_zero", 0), ("player_one", 1)] {
            let penalties_tag = format!("game/{player}_average_penalties");
            self.writer.add_scalar(
                &penalties_tag,
                metrics.average_penalties_per_game[values],
                step,
            );
            let bonus_points_tag = format!("game/{player}_average_bonus_points");
            self.writer.add_scalar(
                &bonus_points_tag,
                metrics.average_bonus_points_per_game[values],
                step,
            );
            let rows_tag = format!("game/{player}_average_rows_filled");
            self.writer.add_scalar(
                &rows_tag,
                metrics.average_rows_filled_per_game[values],
                step,
            );
            let columns_tag = format!("game/{player}_average_columns_filled");
            self.writer.add_scalar(
                &columns_tag,
                metrics.average_columns_filled_per_game[values],
                step,
            );
            let tile_bonuses_tag = format!("game/{player}_average_tile_bonuses");
            self.writer.add_scalar(
                &tile_bonuses_tag,
                metrics.average_tile_bonuses_per_game[values],
                step,
            );
        }
        self.writer.flush();

        println!(
            "iteration={} timesteps={} actor_loss={:.4} critic_loss={:.4} actor_grad={:.3} critic_grad={:.3} actor_update={:.5} critic_update={:.5} return={:.2}+-{:.2} value={:.2}+-{:.2} advantage={:.2}+-{:.2} explained_variance={:.3} score_diff={:.2} win_rate={:.3}",
            metrics.iteration,
            metrics.timesteps,
            metrics.actor_loss,
            metrics.critic_loss,
            metrics.actor_grad_norm,
            metrics.critic_grad_norm,
            metrics.actor_update_norm,
            metrics.critic_update_norm,
            metrics.return_mean,
            metrics.return_std,
            metrics.value_mean,
            metrics.value_std,
            metrics.advantage_mean,
            metrics.advantage_std,
            metrics.explained_variance,
            metrics.mean_final_score_difference,
            metrics.player_zero_win_rate,
        );
    }
}
