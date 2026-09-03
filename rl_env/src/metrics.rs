use crate::ppo::{EpisodeStats, RolloutDiagnostics};

/// Scalar diagnostics emitted after each PPO rollout and update iteration.
#[derive(Debug, Clone, Copy)]
pub struct PpoMetrics {
    pub iteration: usize,
    pub timesteps: usize,
    pub batch_timesteps: usize,
    pub episodes: usize,
    pub mean_episode_return: f32,
    pub mean_episode_length: f32,
    pub mean_final_score_difference: f32,
    pub mean_winner_score: f32,
    pub player_zero_win_rate: f32,
    pub average_penalties_per_game: [f32; 2],
    pub average_bonus_points_per_game: [f32; 2],
    pub average_rows_filled_per_game: [f32; 2],
    pub average_columns_filled_per_game: [f32; 2],
    pub average_tile_bonuses_per_game: [f32; 2],
    pub actor_loss: f32,
    pub critic_loss: f32,
    pub actor_grad_norm: f32,
    pub critic_grad_norm: f32,
    pub actor_grad_clip_coefficient: f32,
    pub critic_grad_clip_coefficient: f32,
    pub approx_kl: f32,
    pub clip_fraction: f32,
    pub entropy: f32,
    pub return_mean: f32,
    pub return_std: f32,
    pub return_min: f32,
    pub return_max: f32,
    pub value_mean: f32,
    pub value_std: f32,
    pub value_min: f32,
    pub value_max: f32,
    pub advantage_mean: f32,
    pub advantage_std: f32,
    pub advantage_min: f32,
    pub advantage_max: f32,
    pub explained_variance: f32,
    pub actor_update_norm: f32,
    pub critic_update_norm: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OptimizationMetrics {
    pub(crate) actor_loss: f32,
    pub(crate) critic_loss: f32,
    pub(crate) actor_grad_norm: f32,
    pub(crate) critic_grad_norm: f32,
    pub(crate) actor_grad_clip_coefficient: f32,
    pub(crate) critic_grad_clip_coefficient: f32,
    pub(crate) approx_kl: f32,
    pub(crate) clip_fraction: f32,
    pub(crate) entropy: f32,
    pub(crate) actor_update_norm: f32,
    pub(crate) critic_update_norm: f32,
}

impl PpoMetrics {
    pub(crate) fn from_update(
        iteration: usize,
        timesteps: usize,
        batch_timesteps: usize,
        episodes: &[EpisodeStats],
        diagnostics: &RolloutDiagnostics,
        optimization: OptimizationMetrics,
    ) -> Self {
        Self {
            iteration,
            timesteps,
            batch_timesteps,
            episodes: episodes.len(),
            mean_episode_return: mean_episode_metric(episodes, |episode| episode.reward_sum),
            mean_episode_length: mean_episode_metric(episodes, |episode| episode.length as f32),
            mean_final_score_difference: mean_episode_metric(episodes, |episode| {
                episode.final_score_difference
            }),
            mean_winner_score: mean_episode_metric(episodes, |episode| episode.winner_score),
            player_zero_win_rate: terminal_win_rate(episodes),
            average_penalties_per_game: mean_episode_array(episodes, |episode| episode.penalties),
            average_bonus_points_per_game: mean_episode_array(episodes, |episode| {
                episode.bonus_points
            }),
            average_rows_filled_per_game: mean_episode_array(episodes, |episode| {
                episode.rows_filled
            }),
            average_columns_filled_per_game: mean_episode_array(episodes, |episode| {
                episode.columns_filled
            }),
            average_tile_bonuses_per_game: mean_episode_array(episodes, |episode| {
                episode.tile_bonuses
            }),
            actor_loss: optimization.actor_loss,
            critic_loss: optimization.critic_loss,
            actor_grad_norm: optimization.actor_grad_norm,
            critic_grad_norm: optimization.critic_grad_norm,
            actor_grad_clip_coefficient: optimization.actor_grad_clip_coefficient,
            critic_grad_clip_coefficient: optimization.critic_grad_clip_coefficient,
            approx_kl: optimization.approx_kl,
            clip_fraction: optimization.clip_fraction,
            entropy: optimization.entropy,
            return_mean: diagnostics.return_stats[0],
            return_std: diagnostics.return_stats[1],
            return_min: diagnostics.return_stats[2],
            return_max: diagnostics.return_stats[3],
            value_mean: diagnostics.value_stats[0],
            value_std: diagnostics.value_stats[1],
            value_min: diagnostics.value_stats[2],
            value_max: diagnostics.value_stats[3],
            advantage_mean: diagnostics.advantage_stats[0],
            advantage_std: diagnostics.advantage_stats[1],
            advantage_min: diagnostics.advantage_stats[2],
            advantage_max: diagnostics.advantage_stats[3],
            explained_variance: diagnostics.explained_variance,
            actor_update_norm: optimization.actor_update_norm,
            critic_update_norm: optimization.critic_update_norm,
        }
    }
}

fn mean_episode_metric<F>(episodes: &[EpisodeStats], metric: F) -> f32
where
    F: Fn(&EpisodeStats) -> f32,
{
    if episodes.is_empty() {
        return 0.0;
    }
    episodes.iter().map(metric).sum::<f32>() / episodes.len() as f32
}

fn mean_episode_array<F>(episodes: &[EpisodeStats], metric: F) -> [f32; 2]
where
    F: Fn(&EpisodeStats) -> [usize; 2],
{
    if episodes.is_empty() {
        return [0.0; 2];
    }
    let totals = episodes
        .iter()
        .map(metric)
        .fold([0.0; 2], |mut totals, values| {
            totals[0] += values[0] as f32;
            totals[1] += values[1] as f32;
            totals
        });
    [
        totals[0] / episodes.len() as f32,
        totals[1] / episodes.len() as f32,
    ]
}

fn terminal_win_rate(episodes: &[EpisodeStats]) -> f32 {
    let terminal_episodes = episodes.iter().filter(|episode| episode.terminated);
    let terminal_count = terminal_episodes.clone().count();
    if terminal_count == 0 {
        return 0.0;
    }
    terminal_episodes
        .filter(|episode| episode.player_zero_won)
        .count() as f32
        / terminal_count as f32
}
