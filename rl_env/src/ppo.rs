//! Minimal on-policy PPO training for the discrete Azul environment.

use std::path::Path;

use rand::RngExt;
use tch::{Kind, Reduction, Tensor, nn, nn::Module, nn::OptimizerConfig, no_grad};

use crate::metrics::{OptimizationMetrics, PpoMetrics};
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
    /// Adam learning rate used by actor.
    pub actor_learning_rate: f64,
    /// Adam learning rate used by critic.
    pub critic_learning_rate: f64,
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

/// A trained actor loaded for inference without PPO optimizer state.
pub struct ActorPolicy {
    var_store: nn::VarStore,
    actor: ActionConditionedActor,
}

impl ActorPolicy {
    /// Loads actor weights from a LibTorch var-store checkpoint.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, tch::TchError> {
        let mut var_store = nn::VarStore::new(get_device());
        let actor = initialize_actor(&var_store.root());
        var_store.load(path)?;
        Ok(Self { var_store, actor })
    }

    /// Saves this actor's weights to a LibTorch var-store checkpoint.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), tch::TchError> {
        self.var_store.save(path)
    }

    /// Scores a batch of states and candidate action features.
    pub fn forward(&self, states: &Tensor, action_features: &Tensor) -> Tensor {
        no_grad(|| self.actor.forward(states, action_features))
    }
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            timesteps_per_batch: 1_000,
            max_timesteps_per_episode: 1_000,
            updates_per_iteration: 4,
            actor_learning_rate: 3e-4,
            critic_learning_rate: 1e-4,
            max_grad_norm: 0.5,
            gamma: 0.99,
            lambda: 0.95,
            lower_clip_epsilon: 0.2,
            upper_clip_epsilon: 0.28,
        }
    }
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
    learner_action: bool,
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
pub(crate) struct EpisodeStats {
    pub(crate) reward_sum: f32,
    pub(crate) length: usize,
    pub(crate) final_score_difference: f32,
    pub(crate) winner_score: f32,
    pub(crate) terminated: bool,
    pub(crate) learner_won: bool,
    pub(crate) penalties: [usize; 2],
    pub(crate) bonus_points: [usize; 2],
    pub(crate) rows_filled: [usize; 2],
    pub(crate) columns_filled: [usize; 2],
    pub(crate) tile_bonuses: [usize; 2],
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
        let learner_actions: Vec<_> = self
            .steps
            .iter()
            .map(|step| if step.learner_action { 1.0 } else { 0.0 })
            .collect();
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
        let learner_action_mask = Tensor::from_slice(&learner_actions).to_device(get_device());
        let action_positions = Tensor::from_slice(&action_positions).to_device(get_device());
        let old_log_probs = Tensor::from_slice(
            &self
                .steps
                .iter()
                .map(|step| step.old_log_prob)
                .collect::<Vec<_>>(),
        )
        .to_device(get_device());

        let raw_advantage_values = compute_gae(
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
        let return_values: Vec<_> = values
            .iter()
            .zip(&raw_advantage_values)
            .map(|(value, advantage)| value + advantage)
            .collect();
        let diagnostics = RolloutDiagnostics {
            return_stats: slice_stats(&return_values),
            value_stats: slice_stats(&values),
            advantage_stats: slice_stats(&raw_advantage_values),
            explained_variance: explained_variance(&values, &return_values),
        };
        let values = Tensor::from_slice(&values).to_device(get_device());
        let raw_advantages = Tensor::from_slice(&raw_advantage_values).to_device(get_device());
        let returns = &raw_advantages + &values;
        let learner_count = learner_action_mask.sum(Kind::Float).clamp_min(1.0);
        let learner_mean =
            (&raw_advantages * &learner_action_mask).sum(Kind::Float) / &learner_count;
        let centered_advantages = &raw_advantages - &learner_mean;
        let learner_std = (centered_advantages.square() * &learner_action_mask)
            .sum(Kind::Float)
            .divide(&learner_count)
            .sqrt();
        let advantages = centered_advantages / (learner_std + 1e-8);

        RolloutData {
            states,
            candidate_features,
            candidate_mask,
            learner_action_mask,
            action_positions,
            old_log_probs,
            returns,
            advantages,
            diagnostics,
        }
    }
}

/// Tensor representation of one on-policy rollout.
struct RolloutData {
    states: Tensor,
    candidate_features: Tensor,
    candidate_mask: Tensor,
    learner_action_mask: Tensor,
    action_positions: Tensor,
    old_log_probs: Tensor,
    returns: Tensor,
    advantages: Tensor,
    diagnostics: RolloutDiagnostics,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RolloutDiagnostics {
    pub(crate) return_stats: [f32; 4],
    pub(crate) value_stats: [f32; 4],
    pub(crate) advantage_stats: [f32; 4],
    pub(crate) explained_variance: f32,
}

fn slice_stats(values: &[f32]) -> [f32; 4] {
    assert!(!values.is_empty());
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    [
        mean,
        variance.sqrt(),
        values.iter().copied().fold(f32::INFINITY, f32::min),
        values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    ]
}

fn explained_variance(values: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(values.len(), targets.len());
    let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let target_variance = targets
        .iter()
        .map(|target| (target - target_mean).powi(2))
        .sum::<f32>()
        / targets.len() as f32;
    if target_variance == 0.0 {
        return 0.0;
    }
    let residual_variance = values
        .iter()
        .zip(targets)
        .map(|(value, target)| (target - value).powi(2))
        .sum::<f32>()
        / targets.len() as f32;
    1.0 - residual_variance / target_variance
}

/// Applies the candidate mask and returns log-probabilities over candidates.
fn masked_log_probs(logits: &Tensor, candidate_mask: &Tensor) -> Tensor {
    let invalid_actions = candidate_mask.eq(0.0);
    logits
        .masked_fill(&invalid_actions, f64::NEG_INFINITY)
        .log_softmax(-1, Kind::Float)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpponentKind {
    Current,
    Historical(usize),
    Random,
    Heuristic,
}

struct HistoricalPolicy {
    _var_store: nn::VarStore,
    actor: ActionConditionedActor,
}

#[derive(Default)]
struct OpponentPool {
    historical: Vec<HistoricalPolicy>,
}

impl OpponentPool {
    fn sample(&self) -> OpponentKind {
        let mut rng = rand::rng();
        let draw = rng.random::<f64>();
        self.kind_for_draw(draw, || rng.random_range(0..self.historical.len()))
    }

    fn kind_for_draw<F>(&self, draw: f64, historical_index: F) -> OpponentKind
    where
        F: FnOnce() -> usize,
    {
        let (current_weight, historical_weight) = if self.historical.is_empty() {
            (0.9, 0.0)
        } else {
            (0.5, 0.4)
        };
        if draw < current_weight {
            OpponentKind::Current
        } else if draw < current_weight + historical_weight {
            OpponentKind::Historical(historical_index())
        } else if draw < current_weight + historical_weight + 0.05 {
            OpponentKind::Random
        } else {
            OpponentKind::Heuristic
        }
    }
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

fn evaluate_action(
    actor: &ActionConditionedActor,
    critic: &ResNetwork,
    state: &Tensor,
    legal_candidates: &[(usize, [f32; ACTION_FEATURE_SIZE])],
    action: usize,
) -> (f32, f32) {
    no_grad(|| {
        let action_index = legal_candidates
            .iter()
            .position(|(candidate, _)| *candidate == action)
            .expect("selected action must be legal");
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
        let log_probs = masked_log_probs(&actor.forward(&state, &action_features), &candidate_mask);
        let action_tensor = Tensor::from_slice(&[action_index as i64])
            .reshape([1, 1])
            .to_device(get_device());
        let old_log_prob = log_probs
            .gather(1, &action_tensor, false)
            .squeeze_dim(1)
            .double_value(&[0]) as f32;
        let value = critic.forward(&state).squeeze_dim(1).double_value(&[0]) as f32;
        (old_log_prob, value)
    })
}

fn select_random_action(legal_candidates: &[(usize, [f32; ACTION_FEATURE_SIZE])]) -> usize {
    rand::rng().random_range(0..legal_candidates.len())
}

fn select_heuristic_action(legal_candidates: &[(usize, [f32; ACTION_FEATURE_SIZE])]) -> usize {
    let dynamic_offset = ACTION_FEATURE_SIZE - 8;
    legal_candidates
        .iter()
        .enumerate()
        .max_by(|(_, (_, left)), (_, (_, right))| {
            let score = |features: &[f32; ACTION_FEATURE_SIZE]| {
                features[dynamic_offset + 7] * 4.0
                    + features[dynamic_offset + 1] * 2.0
                    + features[dynamic_offset + 4]
                    - features[dynamic_offset + 6] * 3.0
                    - features[dynamic_offset + 5] * 2.0
            };
            score(left)
                .partial_cmp(&score(right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .expect("the environment returned no legal actions")
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
    opponent_pool: OpponentPool,
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
        let actor_optimizer = nn::Adam::default().build(&actor_vs, config.actor_learning_rate)?;
        let critic_optimizer =
            nn::Adam::default().build(&critic_vs, config.critic_learning_rate)?;

        Ok(Self {
            actor_vs,
            critic_vs,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            config,
            opponent_pool: OpponentPool::default(),
        })
    }

    /// Saves the current actor as a frozen historical opponent.
    pub fn add_historical_opponent(&mut self) -> Result<(), tch::TchError> {
        let mut var_store = nn::VarStore::new(get_device());
        let actor = initialize_actor(&var_store.root());
        var_store.copy(&self.actor_vs)?;
        self.opponent_pool.historical.push(HistoricalPolicy {
            _var_store: var_store,
            actor,
        });
        Ok(())
    }

    /// Returns the number of frozen historical opponents in the pool.
    pub fn historical_opponent_count(&self) -> usize {
        self.opponent_pool.historical.len()
    }

    /// Saves the current actor and critic weights to separate checkpoints.
    pub fn save_checkpoints<A, C>(&self, actor_path: A, critic_path: C) -> Result<(), tch::TchError>
    where
        A: AsRef<Path>,
        C: AsRef<Path>,
    {
        self.actor_vs.save(actor_path)?;
        self.critic_vs.save(critic_path)
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

            let mut optimization = OptimizationMetrics {
                actor_grad_clip_coefficient: 1.0,
                critic_grad_clip_coefficient: 1.0,
                ..OptimizationMetrics::default()
            };
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
                let actor_objective = unclipped.minimum(&clipped);
                let learner_count = data.learner_action_mask.sum(Kind::Float).clamp_min(1.0);
                let actor_loss_tensor = -(&actor_objective * &data.learner_action_mask)
                    .sum(Kind::Float)
                    / &learner_count;

                let values = self.critic.forward(&data.states).squeeze_dim(1);
                let critic_loss_tensor = values.mse_loss(&data.returns, Reduction::Mean);

                optimization.actor_loss = actor_loss_tensor.double_value(&[]) as f32;
                optimization.critic_loss = critic_loss_tensor.double_value(&[]) as f32;
                optimization.approx_kl = ((&data.old_log_probs - &current_log_probs)
                    * &data.learner_action_mask)
                    .sum(Kind::Float)
                    .divide(&learner_count)
                    .double_value(&[]) as f32;
                optimization.clip_fraction = ratio
                    .lt(1.0 - self.config.lower_clip_epsilon)
                    .logical_or(&ratio.gt(1.0 + self.config.upper_clip_epsilon))
                    .to_kind(Kind::Float)
                    .multiply(&data.learner_action_mask)
                    .sum(Kind::Float)
                    .divide(&learner_count)
                    .double_value(&[]) as f32;
                let finite_log_probs = log_probs.masked_fill(&data.candidate_mask.eq(0.0), 0.0);
                let entropy_per_state = -(&log_probs.exp() * &finite_log_probs).sum_dim_intlist(
                    [-1].as_ref(),
                    false,
                    Kind::Float,
                );
                optimization.entropy = (&entropy_per_state * &data.learner_action_mask)
                    .sum(Kind::Float)
                    .divide(&learner_count)
                    .double_value(&[]) as f32;
                let legal_action_count =
                    data.candidate_mask
                        .sum_dim_intlist([-1].as_ref(), false, Kind::Float);
                optimization.normalized_entropy = (entropy_per_state
                    / legal_action_count.clamp_min(2.0).log())
                .mean(Kind::Float)
                .double_value(&[]) as f32;

                self.actor_optimizer.zero_grad();
                actor_loss_tensor.backward();
                optimization.actor_grad_norm = gradient_norm(&self.actor_vs);
                optimization.actor_grad_clip_coefficient = gradient_clip_coefficient(
                    optimization.actor_grad_norm,
                    self.config.max_grad_norm,
                );
                let actor_parameters = parameter_snapshot(&self.actor_vs);
                self.actor_optimizer
                    .clip_grad_norm(self.config.max_grad_norm);
                self.actor_optimizer.step();
                optimization.actor_update_norm =
                    parameter_update_norm(&actor_parameters, &self.actor_vs);

                self.critic_optimizer.zero_grad();
                critic_loss_tensor.backward();
                optimization.critic_grad_norm = gradient_norm(&self.critic_vs);
                optimization.critic_grad_clip_coefficient = gradient_clip_coefficient(
                    optimization.critic_grad_norm,
                    self.config.max_grad_norm,
                );
                let critic_parameters = parameter_snapshot(&self.critic_vs);
                self.critic_optimizer
                    .clip_grad_norm(self.config.max_grad_norm);
                self.critic_optimizer.step();
                optimization.critic_update_norm =
                    parameter_update_norm(&critic_parameters, &self.critic_vs);
            }

            iteration += 1;
            let metrics = PpoMetrics::from_update(
                iteration,
                timesteps,
                batch_timesteps,
                &episode_stats,
                &data.diagnostics,
                optimization,
            );
            on_update(&metrics);
        }
    }

    /// Collects complete episodes until the configured batch size is reached.
    fn collect_rollout(&self, env: &mut AzulEnv) -> RolloutBatch {
        let mut batch = RolloutBatch::with_capacity(self.config.timesteps_per_batch);
        while batch.len() < self.config.timesteps_per_batch {
            // Rollouts are episode-complete; do not impose a mid-game action cap.
            let mut state = env.reset(usize::MAX);
            let learner_player = rand::rng().random_range(0..2);
            let opponent = self.opponent_pool.sample();
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
                let learner_action = player == learner_player;
                let (action, old_log_prob, value) = if learner_action {
                    let (action, old_log_prob, value) =
                        sample_action(&self.actor, &self.critic, &state, &legal_candidates);
                    (action as usize, old_log_prob, value)
                } else {
                    match opponent {
                        OpponentKind::Current => {
                            let (action, old_log_prob, value) =
                                sample_action(&self.actor, &self.critic, &state, &legal_candidates);
                            (action as usize, old_log_prob, value)
                        }
                        OpponentKind::Historical(index) => {
                            let (action, old_log_prob, value) = sample_action(
                                &self.opponent_pool.historical[index].actor,
                                &self.critic,
                                &state,
                                &legal_candidates,
                            );
                            (action as usize, old_log_prob, value)
                        }
                        OpponentKind::Random => {
                            let action =
                                legal_candidates[select_random_action(&legal_candidates)].0;
                            let (old_log_prob, value) = evaluate_action(
                                &self.actor,
                                &self.critic,
                                &state,
                                &legal_candidates,
                                action,
                            );
                            (action, old_log_prob, value)
                        }
                        OpponentKind::Heuristic => {
                            let action =
                                legal_candidates[select_heuristic_action(&legal_candidates)].0;
                            let (old_log_prob, value) = evaluate_action(
                                &self.actor,
                                &self.critic,
                                &state,
                                &legal_candidates,
                                action,
                            );
                            (action, old_log_prob, value)
                        }
                    }
                };
                let result = env.step(action).expect("masked action must be legal");
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
                    action: action as i64,
                    old_log_prob,
                    reward: result.reward,
                    value,
                    next_value,
                    player,
                    next_player,
                    learner_action,
                    terminated: result.terminated,
                    truncated: result.truncated,
                });
                state = result.next_state;

                if result.terminated {
                    let boards = env.gamestate.get_boards();
                    let opponent_player = 1 - learner_player;
                    let final_score_difference = boards[learner_player].get_score() as f32
                        - boards[opponent_player].get_score() as f32;
                    let winner_score = boards[learner_player]
                        .get_score()
                        .max(boards[opponent_player].get_score())
                        as f32;
                    batch.episodes.push(EpisodeStats {
                        reward_sum: episode_return,
                        length: episode_length,
                        final_score_difference,
                        winner_score,
                        terminated: result.terminated,
                        learner_won: result.terminated
                            && env.gamestate.get_winner() == learner_player,
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

fn parameter_snapshot(var_store: &nn::VarStore) -> Vec<Tensor> {
    no_grad(|| {
        var_store
            .trainable_variables()
            .into_iter()
            .map(|variable| variable.detach().copy())
            .collect()
    })
}

fn parameter_update_norm(before: &[Tensor], var_store: &nn::VarStore) -> f32 {
    no_grad(|| {
        let after = var_store.trainable_variables();
        assert_eq!(before.len(), after.len());
        let squared_norms: Vec<_> = before
            .iter()
            .zip(after)
            .map(|(before, after)| (after - before).square().sum(Kind::Float))
            .collect();
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
    use super::{
        ActorPolicy, AzulEnv, HistoricalPolicy, OpponentKind, OpponentPool, PpoConfig, PpoTrainer,
        compute_gae,
    };
    use crate::{get_device, net::initialize_actor};
    use std::path::PathBuf;
    use tch::{Kind, Tensor, nn};

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
    fn opponent_pool_uses_league_weights_when_history_exists() {
        let var_store = nn::VarStore::new(get_device());
        let actor = initialize_actor(&var_store.root());
        let pool = OpponentPool {
            historical: vec![HistoricalPolicy {
                _var_store: var_store,
                actor,
            }],
        };
        assert_eq!(pool.kind_for_draw(0.49, || 0), OpponentKind::Current);
        assert_eq!(pool.kind_for_draw(0.50, || 0), OpponentKind::Historical(0));
        assert_eq!(pool.kind_for_draw(0.90, || 0), OpponentKind::Random);
        assert_eq!(pool.kind_for_draw(0.96, || 0), OpponentKind::Heuristic);
    }

    #[test]
    fn opponent_pool_falls_back_to_current_for_missing_history() {
        let pool = OpponentPool::default();
        assert_eq!(
            pool.kind_for_draw(0.89, || unreachable!()),
            OpponentKind::Current
        );
        assert_eq!(
            pool.kind_for_draw(0.90, || unreachable!()),
            OpponentKind::Random
        );
        assert_eq!(
            pool.kind_for_draw(0.96, || unreachable!()),
            OpponentKind::Heuristic
        );
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

    #[test]
    fn trainer_can_capture_a_historical_opponent() {
        let mut trainer = PpoTrainer::new(PpoConfig::default()).expect("trainer should initialize");
        trainer
            .add_historical_opponent()
            .expect("historical copy should initialize");
        assert_eq!(trainer.historical_opponent_count(), 1);
    }

    #[test]
    fn trainer_checkpoints_can_be_loaded_for_inference() {
        let mut actor_path = PathBuf::from(std::env::temp_dir());
        actor_path.push(format!("azul-actor-{}.ot", std::process::id()));
        let mut critic_path = PathBuf::from(std::env::temp_dir());
        critic_path.push(format!("azul-critic-{}.ot", std::process::id()));

        let trainer = PpoTrainer::new(PpoConfig::default()).expect("trainer should initialize");
        trainer
            .save_checkpoints(&actor_path, &critic_path)
            .expect("checkpoints should save");
        ActorPolicy::load(&actor_path).expect("actor checkpoint should load");

        std::fs::remove_file(actor_path).expect("actor checkpoint should be removable");
        std::fs::remove_file(critic_path).expect("critic checkpoint should be removable");
    }
}
