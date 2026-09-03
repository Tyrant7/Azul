//! Minimal on-policy PPO training for the discrete Azul environment.

use tch::{Kind, Reduction, Tensor, nn, nn::Module, nn::OptimizerConfig, no_grad};

use crate::net::{ActionConditionedActor, ResNetwork, initialize_actor, initialize_critic};
use crate::{ACTION_FEATURE_SIZE, AzulEnv, get_device};

/// Hyperparameters for the minimal PPO trainer.
#[derive(Debug, Clone, Copy)]
pub struct PpoConfig {
    /// Number of environment transitions collected before each update.
    pub timesteps_per_batch: usize,
    /// Maximum number of transitions in one episode.
    pub max_timesteps_per_episode: usize,
    /// Number of full-batch optimization passes over each rollout.
    pub updates_per_iteration: usize,
    /// Adam learning rate used by both actor and critic.
    pub learning_rate: f64,
    /// Maximum global L2 norm applied to actor and critic gradients.
    pub max_grad_norm: f64,
    /// Discount factor used for return and advantage estimates.
    pub gamma: f64,
    /// GAE lambda smoothing parameter.
    pub lambda: f64,
    /// PPO probability-ratio clipping range for disadvantaged actions.
    pub lower_clip_epsilon: f64,
    /// PPO probability-ratio clipping range for advantaged actions.
    pub upper_clip_epsilon: f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            timesteps_per_batch: 1_000,
            max_timesteps_per_episode: 1_000,
            updates_per_iteration: 4,
            learning_rate: 3e-4,
            max_grad_norm: 0.5,
            gamma: 0.99,
            lambda: 0.95,
            lower_clip_epsilon: 0.2,
            upper_clip_epsilon: 0.28,
        }
    }
}

/// Scalar diagnostics emitted after each PPO rollout and update iteration.
#[derive(Debug, Clone, Copy)]
pub struct PpoMetrics {
    /// One-based PPO iteration number.
    pub iteration: usize,
    /// Total environment transitions collected so far.
    pub timesteps: usize,
    /// Number of transitions in this rollout batch.
    pub batch_timesteps: usize,
    /// Number of complete games in this batch.
    pub episodes: usize,
    /// Mean sum of environment rewards per episode.
    pub mean_episode_return: f32,
    /// Mean number of transitions per episode.
    pub mean_episode_length: f32,
    /// Mean final player-zero-minus-player-one score difference.
    pub mean_final_score_difference: f32,
    /// Mean winner score.
    pub mean_winner_score: f32,
    /// Fraction of terminal episodes won by player zero.
    pub player_zero_win_rate: f32,
    /// Mean scoring-space penalty tiles per game for each player.
    pub average_penalties_per_game: [f32; 2],
    /// Mean bonus points earned per game for each player.
    pub average_bonus_points_per_game: [f32; 2],
    /// Mean wall rows completed per game for each player.
    pub average_rows_filled_per_game: [f32; 2],
    /// Mean wall columns completed per game for each player.
    pub average_columns_filled_per_game: [f32; 2],
    /// Mean five-of-a-kind bonuses earned per game for each player.
    pub average_tile_bonuses_per_game: [f32; 2],
    /// Actor surrogate loss from the final update epoch.
    pub actor_loss: f32,
    /// Critic mean-squared error from the final update epoch.
    pub critic_loss: f32,
    /// Actor global gradient norm measured before clipping in the final epoch.
    pub actor_grad_norm: f32,
    /// Critic global gradient norm measured before clipping in the final epoch.
    pub critic_grad_norm: f32,
    /// Actor gradient scaling coefficient applied by global-norm clipping.
    pub actor_grad_clip_coefficient: f32,
    /// Critic gradient scaling coefficient applied by global-norm clipping.
    pub critic_grad_clip_coefficient: f32,
    /// Approximate KL divergence from the rollout policy in the final epoch.
    pub approx_kl: f32,
    /// Fraction of samples outside the PPO clipping range in the final epoch.
    pub clip_fraction: f32,
}

/// A single transition collected under the current policy.
struct RolloutStep {
    state: Tensor,
    legal_actions: Vec<i64>,
    action_features: Vec<[f32; ACTION_FEATURE_SIZE]>,
    action: i64,
    old_log_prob: f32,
    reward: f32,
    value: f32,
    next_value: f32,
    player: usize,
    next_player: usize,
    terminated: bool,
    truncated: bool,
}

/// Temporary on-policy data collected before one PPO update iteration.
struct RolloutBatch {
    steps: Vec<RolloutStep>,
    episodes: Vec<EpisodeStats>,
}

/// Episode-level values retained for training diagnostics.
#[derive(Debug, Clone, Copy)]
struct EpisodeStats {
    reward_sum: f32,
    length: usize,
    final_score_difference: f32,
    winner_score: f32,
    terminated: bool,
    player_zero_won: bool,
    penalties: [usize; 2],
    bonus_points: [usize; 2],
    rows_filled: [usize; 2],
    columns_filled: [usize; 2],
    tile_bonuses: [usize; 2],
}

impl RolloutBatch {
    /// Creates an empty rollout batch with the requested capacity.
    fn with_capacity(capacity: usize) -> Self {
        Self {
            steps: Vec::with_capacity(capacity),
            episodes: Vec::new(),
        }
    }

    /// Returns the number of collected transitions.
    fn len(&self) -> usize {
        self.steps.len()
    }

    /// Converts rollout steps into tensors and computes normalized advantages.
    fn into_data(self, gamma: f64, lambda: f64) -> RolloutData {
        let rewards: Vec<_> = self.steps.iter().map(|step| step.reward).collect();
        let players: Vec<_> = self.steps.iter().map(|step| step.player).collect();
        let next_players: Vec<_> = self.steps.iter().map(|step| step.next_player).collect();
        let terminated: Vec<_> = self.steps.iter().map(|step| step.terminated).collect();
        let truncated: Vec<_> = self.steps.iter().map(|step| step.truncated).collect();
        let values: Vec<_> = self.steps.iter().map(|step| step.value).collect();
        let next_values: Vec<_> = self.steps.iter().map(|step| step.next_value).collect();

        let states = Tensor::stack(
            &self
                .steps
                .iter()
                .map(|step| step.state.shallow_clone())
                .collect::<Vec<_>>(),
            0,
        );
        let max_legal_actions = self
            .steps
            .iter()
            .map(|step| step.legal_actions.len())
            .max()
            .expect("rollout batches must contain at least one step");
        let mut candidate_features =
            vec![0.0_f32; self.steps.len() * max_legal_actions * ACTION_FEATURE_SIZE];
        let mut candidate_mask = vec![0.0_f32; self.steps.len() * max_legal_actions];
        let mut action_positions = Vec::with_capacity(self.steps.len());
        for (step_index, step) in self.steps.iter().enumerate() {
            let selected_position = step
                .legal_actions
                .iter()
                .position(|&action| action == step.action)
                .expect("sampled action must be in the legal candidate list");
            action_positions.push(selected_position as i64);
            for (candidate_index, features) in step.action_features.iter().enumerate() {
                let feature_offset =
                    (step_index * max_legal_actions + candidate_index) * ACTION_FEATURE_SIZE;
                candidate_features[feature_offset..feature_offset + ACTION_FEATURE_SIZE]
                    .copy_from_slice(features);
                candidate_mask[step_index * max_legal_actions + candidate_index] = 1.0;
            }
        }
        let candidate_features = Tensor::from_slice(&candidate_features)
            .reshape([
                self.steps.len() as i64,
                max_legal_actions as i64,
                ACTION_FEATURE_SIZE as i64,
            ])
            .to_device(get_device());
        let candidate_mask = Tensor::from_slice(&candidate_mask)
            .reshape([self.steps.len() as i64, max_legal_actions as i64])
            .to_device(get_device());
        let action_positions = Tensor::from_slice(&action_positions).to_device(get_device());
        let old_log_probs = Tensor::from_slice(
            &self
                .steps
                .iter()
                .map(|step| step.old_log_prob)
                .collect::<Vec<_>>(),
        )
        .to_device(get_device());

        let raw_advantages = compute_gae(
            &rewards,
            &values,
            &next_values,
            &players,
            &next_players,
            &terminated,
            &truncated,
            gamma,
            lambda,
        );
        let values = Tensor::from_slice(&values).to_device(get_device());
        let raw_advantages = Tensor::from_slice(&raw_advantages).to_device(get_device());
        let returns = &raw_advantages + &values;
        let advantages = (&raw_advantages - raw_advantages.mean(Kind::Float))
            / (raw_advantages.std(false) + 1e-8);

        RolloutData {
            states,
            candidate_features,
            candidate_mask,
            action_positions,
            old_log_probs,
            returns,
            advantages,
        }
    }
}

/// Tensor representation of one on-policy rollout.
struct RolloutData {
    states: Tensor,
    candidate_features: Tensor,
    candidate_mask: Tensor,
    action_positions: Tensor,
    old_log_probs: Tensor,
    returns: Tensor,
    advantages: Tensor,
}

/// Applies the candidate mask and returns log-probabilities over candidates.
fn masked_log_probs(logits: &Tensor, candidate_mask: &Tensor) -> Tensor {
    let invalid_actions = candidate_mask.eq(0.0);
    logits
        .masked_fill(&invalid_actions, f64::NEG_INFINITY)
        .log_softmax(-1, Kind::Float)
}

/// Samples one legal candidate and records its old log-probability and value estimate.
fn sample_action(
    actor: &ActionConditionedActor,
    critic: &ResNetwork,
    state: &Tensor,
    legal_candidates: &[(usize, [f32; ACTION_FEATURE_SIZE])],
) -> (i64, f32, f32) {
    no_grad(|| {
        assert!(
            !legal_candidates.is_empty(),
            "the environment returned no legal actions"
        );
        let state = state.unsqueeze(0);
        let feature_values: Vec<_> = legal_candidates
            .iter()
            .flat_map(|(_, features)| features.iter().copied())
            .collect();
        let action_features = Tensor::from_slice(&feature_values)
            .reshape([1, legal_candidates.len() as i64, ACTION_FEATURE_SIZE as i64])
            .to_device(get_device());
        let candidate_mask = Tensor::ones(
            [1, legal_candidates.len() as i64],
            (Kind::Float, get_device()),
        );
        let logits = actor.forward(&state, &action_features);
        let log_probs = masked_log_probs(&logits, &candidate_mask);
        let candidate = log_probs.exp().multinomial(1, false);
        let candidate_index = candidate.int64_value(&[0, 0]) as usize;
        let old_log_prob = log_probs
            .gather(1, &candidate, false)
            .squeeze_dim(1)
            .double_value(&[0]) as f32;
        let value = critic.forward(&state).squeeze_dim(1).double_value(&[0]) as f32;
        (
            legal_candidates[candidate_index].0 as i64,
            old_log_prob,
            value,
        )
    })
}

/// Computes the mean of one metric across the episodes in a rollout batch.
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

/// Computes player zero's win rate across terminal, non-truncated episodes.
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

/// Owns the actor, critic, and optimizers for the minimal PPO algorithm.
pub struct PpoTrainer {
    actor_vs: nn::VarStore,
    critic_vs: nn::VarStore,
    actor: ActionConditionedActor,
    critic: ResNetwork,
    actor_optimizer: nn::Optimizer,
    critic_optimizer: nn::Optimizer,
    config: PpoConfig,
}

impl PpoTrainer {
    /// Creates independently parameterized actor and critic networks.
    pub fn new(config: PpoConfig) -> Result<Self, tch::TchError> {
        assert!(config.timesteps_per_batch > 0);
        assert!(config.max_timesteps_per_episode > 0);
        assert!(config.updates_per_iteration > 0);
        assert!(config.max_grad_norm.is_finite() && config.max_grad_norm > 0.0);
        assert!(config.gamma >= 0.0 && config.gamma <= 1.0);
        assert!(config.lambda >= 0.0 && config.lambda <= 1.0);
        assert!(config.lower_clip_epsilon > 0.0);
        assert!(config.upper_clip_epsilon > 0.0);

        let actor_vs = nn::VarStore::new(get_device());
        let critic_vs = nn::VarStore::new(get_device());
        let actor = initialize_actor(&actor_vs.root());
        let critic = initialize_critic(&critic_vs.root());
        let actor_optimizer = nn::Adam::default().build(&actor_vs, config.learning_rate)?;
        let critic_optimizer = nn::Adam::default().build(&critic_vs, config.learning_rate)?;

        Ok(Self {
            actor_vs,
            critic_vs,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            config,
        })
    }

    /// Collects one fresh rollout batch and applies PPO updates to it.
    pub fn train(&mut self, env: &mut AzulEnv, total_timesteps: usize) {
        self.train_with_callback(env, total_timesteps, |_| {});
    }

    /// Trains PPO and reports metrics after each rollout/update iteration.
    pub fn train_with_callback<F>(
        &mut self,
        env: &mut AzulEnv,
        total_timesteps: usize,
        mut on_update: F,
    ) where
        F: FnMut(&PpoMetrics),
    {
        let mut timesteps = 0;
        let mut iteration = 0;
        while timesteps < total_timesteps {
            let batch = self.collect_rollout(env);
            let batch_timesteps = batch.len();
            let episode_stats = batch.episodes.clone();
            timesteps += batch_timesteps;
            let data = batch.into_data(self.config.gamma, self.config.lambda);

            let mut actor_loss = 0.0;
            let mut critic_loss = 0.0;
            let mut actor_grad_norm = 0.0;
            let mut critic_grad_norm = 0.0;
            let mut actor_grad_clip_coefficient = 1.0;
            let mut critic_grad_clip_coefficient = 1.0;
            let mut approx_kl = 0.0;
            let mut clip_fraction = 0.0;
            for _ in 0..self.config.updates_per_iteration {
                let logits = self.actor.forward(&data.states, &data.candidate_features);
                let log_probs = masked_log_probs(&logits, &data.candidate_mask);
                let current_log_probs = log_probs
                    .gather(1, &data.action_positions.unsqueeze(1), false)
                    .squeeze_dim(1);
                let ratio = (&current_log_probs - &data.old_log_probs).exp();
                let unclipped = &ratio * &data.advantages;
                let clipped_ratio = ratio.clamp(
                    1.0 - self.config.lower_clip_epsilon,
                    1.0 + self.config.upper_clip_epsilon,
                );
                let clipped = clipped_ratio * &data.advantages;
                let actor_loss_tensor = -unclipped.minimum(&clipped).mean(Kind::Float);

                let values = self.critic.forward(&data.states).squeeze_dim(1);
                let critic_loss_tensor = values.mse_loss(&data.returns, Reduction::Mean);

                actor_loss = actor_loss_tensor.double_value(&[]);
                critic_loss = critic_loss_tensor.double_value(&[]);
                approx_kl = (&data.old_log_probs - &current_log_probs)
                    .mean(Kind::Float)
                    .double_value(&[]);
                clip_fraction = ratio
                    .lt(1.0 - self.config.lower_clip_epsilon)
                    .logical_or(&ratio.gt(1.0 + self.config.upper_clip_epsilon))
                    .to_kind(Kind::Float)
                    .mean(Kind::Float)
                    .double_value(&[]);

                self.actor_optimizer.zero_grad();
                actor_loss_tensor.backward();
                actor_grad_norm = gradient_norm(&self.actor_vs);
                actor_grad_clip_coefficient =
                    gradient_clip_coefficient(actor_grad_norm, self.config.max_grad_norm);
                self.actor_optimizer
                    .clip_grad_norm(self.config.max_grad_norm);
                self.actor_optimizer.step();

                self.critic_optimizer.zero_grad();
                critic_loss_tensor.backward();
                critic_grad_norm = gradient_norm(&self.critic_vs);
                critic_grad_clip_coefficient =
                    gradient_clip_coefficient(critic_grad_norm, self.config.max_grad_norm);
                self.critic_optimizer
                    .clip_grad_norm(self.config.max_grad_norm);
                self.critic_optimizer.step();
            }

            iteration += 1;
            on_update(&PpoMetrics {
                iteration,
                timesteps,
                batch_timesteps,
                episodes: episode_stats.len(),
                mean_episode_return: mean_episode_metric(&episode_stats, |episode| {
                    episode.reward_sum
                }),
                mean_episode_length: mean_episode_metric(&episode_stats, |episode| {
                    episode.length as f32
                }),
                mean_final_score_difference: mean_episode_metric(&episode_stats, |episode| {
                    episode.final_score_difference
                }),
                mean_winner_score: mean_episode_metric(&episode_stats, |episode| {
                    episode.winner_score
                }),
                player_zero_win_rate: terminal_win_rate(&episode_stats),
                average_penalties_per_game: mean_episode_array(&episode_stats, |episode| {
                    episode.penalties
                }),
                average_bonus_points_per_game: mean_episode_array(&episode_stats, |episode| {
                    episode.bonus_points
                }),
                average_rows_filled_per_game: mean_episode_array(&episode_stats, |episode| {
                    episode.rows_filled
                }),
                average_columns_filled_per_game: mean_episode_array(&episode_stats, |episode| {
                    episode.columns_filled
                }),
                average_tile_bonuses_per_game: mean_episode_array(&episode_stats, |episode| {
                    episode.tile_bonuses
                }),
                actor_loss: actor_loss as f32,
                critic_loss: critic_loss as f32,
                actor_grad_norm,
                critic_grad_norm,
                actor_grad_clip_coefficient,
                critic_grad_clip_coefficient,
                approx_kl: approx_kl as f32,
                clip_fraction: clip_fraction as f32,
            });
        }
    }

    /// Collects complete episodes until the configured batch size is reached.
    fn collect_rollout(&self, env: &mut AzulEnv) -> RolloutBatch {
        let mut batch = RolloutBatch::with_capacity(self.config.timesteps_per_batch);
        while batch.len() < self.config.timesteps_per_batch {
            // Rollouts are episode-complete; do not impose a mid-game action cap.
            let mut state = env.reset(usize::MAX);
            let mut episode_return = 0.0;
            let mut penalties = [0; 2];
            let mut bonus_points = [0; 2];
            let mut rows_filled = [0; 2];
            let mut columns_filled = [0; 2];
            let mut tile_bonuses = [0; 2];
            let mut episode_length = 0;
            loop {
                let legal_candidates = env.legal_action_features();
                let legal_actions = legal_candidates
                    .iter()
                    .map(|(action, _)| *action as i64)
                    .collect();
                let action_features = legal_candidates
                    .iter()
                    .map(|(_, features)| *features)
                    .collect();
                let player = env.gamestate.get_active_player();
                let (action, old_log_prob, value) =
                    sample_action(&self.actor, &self.critic, &state, &legal_candidates);
                let result = env
                    .step(action as usize)
                    .expect("masked action must be legal");
                let next_player = env.gamestate.get_active_player();
                let next_value = if result.terminated {
                    0.0
                } else {
                    no_grad(|| {
                        self.critic
                            .forward(&result.next_state.unsqueeze(0))
                            .squeeze_dim(1)
                            .double_value(&[0]) as f32
                    })
                };
                episode_return += result.reward;
                if let Some(diagnostics) = result.round_diagnostics {
                    for player in 0..2 {
                        penalties[player] += diagnostics.penalties[player];
                        bonus_points[player] += diagnostics.bonus_points[player];
                        rows_filled[player] += diagnostics.rows_filled[player];
                        columns_filled[player] += diagnostics.columns_filled[player];
                        tile_bonuses[player] += diagnostics.tile_bonuses[player];
                    }
                }
                episode_length += 1;

                batch.steps.push(RolloutStep {
                    state,
                    legal_actions,
                    action_features,
                    action,
                    old_log_prob,
                    reward: result.reward,
                    value,
                    next_value,
                    player,
                    next_player,
                    terminated: result.terminated,
                    truncated: result.truncated,
                });
                state = result.next_state;

                if result.terminated {
                    let boards = env.gamestate.get_boards();
                    let final_score_difference =
                        boards[0].get_score() as f32 - boards[1].get_score() as f32;
                    let winner_score = boards[0].get_score().max(boards[1].get_score()) as f32;
                    batch.episodes.push(EpisodeStats {
                        reward_sum: episode_return,
                        length: episode_length,
                        final_score_difference,
                        winner_score,
                        terminated: result.terminated,
                        player_zero_won: result.terminated && env.gamestate.get_winner() == 0,
                        penalties,
                        bonus_points,
                        rows_filled,
                        columns_filled,
                        tile_bonuses,
                    });
                    break;
                }
            }
        }
        batch
    }

    /// Returns the current actor and critic parameter stores for checkpointing.
    pub fn var_stores(&self) -> (&nn::VarStore, &nn::VarStore) {
        (&self.actor_vs, &self.critic_vs)
    }
}

/// Computes the global L2 norm of all defined gradients in a parameter store.
fn gradient_norm(var_store: &nn::VarStore) -> f32 {
    no_grad(|| {
        let squared_norms: Vec<_> = var_store
            .trainable_variables()
            .into_iter()
            .map(|variable| variable.grad())
            .filter(|gradient| gradient.defined())
            .map(|gradient| gradient.square().sum(Kind::Float))
            .collect();
        if squared_norms.is_empty() {
            return 0.0;
        }
        Tensor::stack(&squared_norms, 0)
            .sum(Kind::Float)
            .sqrt()
            .double_value(&[]) as f32
    })
}

/// Returns the global-norm scaling coefficient that clipping will apply.
fn gradient_clip_coefficient(gradient_norm: f32, maximum_norm: f64) -> f32 {
    (maximum_norm / (gradient_norm as f64 + 1e-6)).min(1.0) as f32
}

/// Computes player-relative generalized advantage estimates over a rollout.
fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    next_values: &[f32],
    players: &[usize],
    next_players: &[usize],
    terminated: &[bool],
    truncated: &[bool],
    gamma: f64,
    lambda: f64,
) -> Vec<f32> {
    assert_eq!(rewards.len(), values.len());
    assert_eq!(rewards.len(), next_values.len());
    assert_eq!(rewards.len(), players.len());
    assert_eq!(rewards.len(), next_players.len());
    assert_eq!(rewards.len(), terminated.len());
    assert_eq!(rewards.len(), truncated.len());

    let timesteps = rewards.len();
    let mut advantages = vec![0.0; timesteps];
    let mut last_gae = 0.0;

    for step in (0..timesteps).rev() {
        let perspective = if players[step] == next_players[step] {
            1.0
        } else {
            -1.0
        };
        // A time-limit truncation still has a valid next-state value.
        let bootstrap_mask = if terminated[step] { 0.0 } else { 1.0 };
        // Do not let the trace cross either a terminal state or an environment reset.
        let trace_mask = if terminated[step] || truncated[step] {
            0.0
        } else {
            1.0
        };
        let delta = rewards[step] + gamma as f32 * perspective * bootstrap_mask * next_values[step]
            - values[step];

        last_gae = delta + gamma as f32 * lambda as f32 * perspective * trace_mask * last_gae;
        advantages[step] = last_gae;
    }
    advantages
}

#[cfg(test)]
mod tests {
    use super::{AzulEnv, PpoConfig, PpoTrainer, compute_gae};
    use tch::{Kind, Tensor};

    #[test]
    fn gae_recurses_backward() {
        assert_eq!(
            compute_gae(
                &[1.0, 2.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[0, 0],
                &[0, 0],
                &[false, true],
                &[false, false],
                1.0,
                0.5,
            ),
            vec![2.0, 2.0]
        );
    }

    #[test]
    fn gae_flips_between_two_player_perspectives() {
        assert_eq!(
            compute_gae(
                &[0.0, 0.0, 4.0],
                &[0.0, 0.0, 0.0],
                &[0.0, 0.0, 0.0],
                &[0, 1, 0],
                &[1, 0, 1],
                &[false, false, true],
                &[false, false, false],
                1.0,
                1.0,
            ),
            vec![4.0, -4.0, 4.0]
        );
    }

    #[test]
    fn gae_bootstraps_truncation_without_crossing_episode_boundary() {
        assert_eq!(
            compute_gae(
                &[0.0, 5.0],
                &[0.0, 0.0],
                &[2.0, 0.0],
                &[0, 0],
                &[0, 0],
                &[false, true],
                &[true, false],
                1.0,
                1.0,
            ),
            vec![2.0, 5.0]
        );
    }

    #[test]
    fn masked_log_probs_make_illegal_actions_impossible() {
        let logits = Tensor::from_slice(&[1.0_f32, 2.0, 3.0]).unsqueeze(0);
        let mask = Tensor::from_slice(&[1.0_f32, 0.0, 1.0]).unsqueeze(0);
        let log_probs = super::masked_log_probs(&logits, &mask);

        assert!(log_probs.double_value(&[0, 1]).is_infinite());
        let legal_probability_sum = log_probs.exp().sum(Kind::Float).double_value(&[]);
        assert!((legal_probability_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trainer_collects_a_complete_game_before_updating() {
        let config = PpoConfig {
            timesteps_per_batch: 1,
            max_timesteps_per_episode: 1,
            updates_per_iteration: 1,
            ..PpoConfig::default()
        };
        let mut trainer = PpoTrainer::new(config).expect("trainer should initialize");
        let mut environment = AzulEnv::new(0, config.max_timesteps_per_episode);

        let mut callbacks = 0;
        let mut reported_timesteps = 0;
        trainer.train_with_callback(&mut environment, 1, |metrics| {
            callbacks += 1;
            reported_timesteps = metrics.timesteps;
            assert!(metrics.batch_timesteps > 1);
            assert_eq!(metrics.episodes, 1);
        });

        assert_eq!(callbacks, 1);
        assert!(reported_timesteps > 1);
    }
}
