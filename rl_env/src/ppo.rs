//! Minimal on-policy PPO training for the discrete Azul environment.

use tch::{Kind, Reduction, Tensor, nn, nn::Module, nn::OptimizerConfig, no_grad};

use crate::net::{ResNetwork, initialize_actor, initialize_critic};
use crate::{AzulEnv, get_device};

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
    /// Discount factor used for rewards-to-go.
    pub gamma: f64,
    /// PPO probability-ratio clipping range.
    pub clip: f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            timesteps_per_batch: 1_000,
            max_timesteps_per_episode: 1_000,
            updates_per_iteration: 5,
            learning_rate: 3e-4,
            gamma: 0.99,
            clip: 0.2,
        }
    }
}

/// A single transition collected under the current policy.
struct RolloutStep {
    state: Tensor,
    action_mask: Tensor,
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
}

impl RolloutBatch {
    /// Creates an empty rollout batch with the requested capacity.
    fn with_capacity(capacity: usize) -> Self {
        Self {
            steps: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of collected transitions.
    fn len(&self) -> usize {
        self.steps.len()
    }

    /// Converts rollout steps into tensors and computes normalized advantages.
    fn into_data(self, gamma: f64) -> RolloutData {
        let rewards: Vec<_> = self.steps.iter().map(|step| step.reward).collect();
        let players: Vec<_> = self.steps.iter().map(|step| step.player).collect();
        let next_players: Vec<_> = self.steps.iter().map(|step| step.next_player).collect();
        let terminated: Vec<_> = self.steps.iter().map(|step| step.terminated).collect();
        let truncated: Vec<_> = self.steps.iter().map(|step| step.truncated).collect();
        let next_values: Vec<_> = self.steps.iter().map(|step| step.next_value).collect();
        let rewards_to_go = compute_rewards_to_go(
            &rewards,
            &players,
            &next_players,
            &terminated,
            &truncated,
            &next_values,
            gamma,
        );

        let states = Tensor::stack(
            &self
                .steps
                .iter()
                .map(|step| step.state.shallow_clone())
                .collect::<Vec<_>>(),
            0,
        );
        let action_masks = Tensor::stack(
            &self
                .steps
                .iter()
                .map(|step| step.action_mask.shallow_clone())
                .collect::<Vec<_>>(),
            0,
        );
        let actions = Tensor::from_slice(
            &self
                .steps
                .iter()
                .map(|step| step.action)
                .collect::<Vec<_>>(),
        )
        .to_device(get_device());
        let old_log_probs = Tensor::from_slice(
            &self
                .steps
                .iter()
                .map(|step| step.old_log_prob)
                .collect::<Vec<_>>(),
        )
        .to_device(get_device());
        let values =
            Tensor::from_slice(&self.steps.iter().map(|step| step.value).collect::<Vec<_>>())
                .to_device(get_device());
        let returns = Tensor::from_slice(&rewards_to_go).to_device(get_device());
        let advantages = &returns - &values;
        let advantages =
            (&advantages - advantages.mean(Kind::Float)) / (advantages.std(false) + 1e-8);

        RolloutData {
            states,
            action_masks,
            actions,
            old_log_probs,
            returns,
            advantages,
        }
    }
}

/// Tensor representation of one on-policy rollout.
struct RolloutData {
    states: Tensor,
    action_masks: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    returns: Tensor,
    advantages: Tensor,
}

/// Computes discounted rewards-to-go, accounting for Azul's changing player perspective.
fn compute_rewards_to_go(
    rewards: &[f32],
    players: &[usize],
    next_players: &[usize],
    terminated: &[bool],
    truncated: &[bool],
    next_values: &[f32],
    gamma: f64,
) -> Vec<f32> {
    assert_eq!(rewards.len(), players.len());
    assert_eq!(rewards.len(), next_players.len());
    assert_eq!(rewards.len(), terminated.len());
    assert_eq!(rewards.len(), truncated.len());
    assert_eq!(rewards.len(), next_values.len());

    let mut rewards_to_go = vec![0.0; rewards.len()];
    let mut next_return = 0.0;
    for index in (0..rewards.len()).rev() {
        let continuation = if terminated[index] {
            0.0
        } else if truncated[index] {
            next_values[index]
        } else {
            next_return
        };
        let perspective = if players[index] == next_players[index] {
            1.0
        } else {
            -1.0
        };
        rewards_to_go[index] = rewards[index] + gamma as f32 * perspective * continuation;
        next_return = rewards_to_go[index];
    }
    rewards_to_go
}

/// Applies the legal-action mask and returns log-probabilities over actions.
fn masked_log_probs(logits: &Tensor, action_masks: &Tensor) -> Tensor {
    let invalid_actions = action_masks.eq(0.0);
    logits
        .masked_fill(&invalid_actions, f64::NEG_INFINITY)
        .log_softmax(-1, Kind::Float)
}

/// Samples one masked action and records its old log-probability and value estimate.
fn sample_action(
    actor: &ResNetwork,
    critic: &ResNetwork,
    state: &Tensor,
    action_mask: &Tensor,
) -> (i64, f32, f32) {
    no_grad(|| {
        assert!(
            action_mask.sum(Kind::Float).double_value(&[]) > 0.0,
            "the environment returned no legal actions"
        );
        let state = state.unsqueeze(0);
        let logits = actor.forward(&state);
        let log_probs = masked_log_probs(&logits, &action_mask.unsqueeze(0));
        let action = log_probs.exp().multinomial(1, false);
        let old_log_prob = log_probs
            .gather(1, &action, false)
            .squeeze_dim(1)
            .double_value(&[0]) as f32;
        let value = critic.forward(&state).squeeze_dim(1).double_value(&[0]) as f32;
        (action.int64_value(&[0, 0]), old_log_prob, value)
    })
}

/// Owns the actor, critic, and optimizers for the minimal PPO algorithm.
pub struct PpoTrainer {
    actor_vs: nn::VarStore,
    critic_vs: nn::VarStore,
    actor: ResNetwork,
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
        assert!(config.gamma >= 0.0 && config.gamma <= 1.0);
        assert!(config.clip > 0.0);

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
        let mut timesteps = 0;
        while timesteps < total_timesteps {
            let batch = self.collect_rollout(env);
            timesteps += batch.len();
            let data = batch.into_data(self.config.gamma);

            for _ in 0..self.config.updates_per_iteration {
                let logits = self.actor.forward(&data.states);
                let log_probs = masked_log_probs(&logits, &data.action_masks);
                let current_log_probs = log_probs
                    .gather(1, &data.actions.unsqueeze(1), false)
                    .squeeze_dim(1);
                let ratio = (current_log_probs - &data.old_log_probs).exp();
                let unclipped = &ratio * &data.advantages;
                let clipped_ratio = ratio.clamp(1.0 - self.config.clip, 1.0 + self.config.clip);
                let clipped = clipped_ratio * &data.advantages;
                let actor_loss = -unclipped.minimum(&clipped).mean(Kind::Float);

                let values = self.critic.forward(&data.states).squeeze_dim(1);
                let critic_loss = values.mse_loss(&data.returns, Reduction::Mean);

                self.actor_optimizer.zero_grad();
                actor_loss.backward();
                self.actor_optimizer.step();

                self.critic_optimizer.zero_grad();
                critic_loss.backward();
                self.critic_optimizer.step();
            }
        }
    }

    /// Collects complete episodes until the configured batch size is reached.
    fn collect_rollout(&self, env: &mut AzulEnv) -> RolloutBatch {
        let mut batch = RolloutBatch::with_capacity(self.config.timesteps_per_batch);
        while batch.len() < self.config.timesteps_per_batch {
            let mut state = env.reset(self.config.max_timesteps_per_episode);
            for _ in 0..self.config.max_timesteps_per_episode {
                let action_mask = env.action_mask();
                let player = env.gamestate.get_active_player();
                let (action, old_log_prob, value) =
                    sample_action(&self.actor, &self.critic, &state, &action_mask);
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

                batch.steps.push(RolloutStep {
                    state,
                    action_mask,
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

                if result.terminated || result.truncated {
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

/// Runs a small default PPO training session for manual experiments.
pub fn ppo() {
    let config = PpoConfig::default();
    let mut trainer = PpoTrainer::new(config).expect("PPO networks must initialize");
    let mut environment = AzulEnv::new(rand::random(), config.max_timesteps_per_episode);
    trainer.train(&mut environment, config.timesteps_per_batch);
}

#[cfg(test)]
mod tests {
    use super::{AzulEnv, PpoConfig, PpoTrainer, compute_rewards_to_go};
    use tch::{Kind, Tensor};

    #[test]
    fn rewards_to_go_matches_discounted_returns() {
        assert_eq!(
            compute_rewards_to_go(
                &[1.0, 2.0, 3.0],
                &[0, 0, 0],
                &[0, 0, 0],
                &[false, false, true],
                &[false, false, false],
                &[0.0, 0.0, 0.0],
                1.0,
            ),
            vec![6.0, 5.0, 3.0]
        );
    }

    #[test]
    fn rewards_to_go_flips_between_two_player_perspectives() {
        assert_eq!(
            compute_rewards_to_go(
                &[0.0, 0.0, 4.0],
                &[0, 1, 0],
                &[1, 0, 1],
                &[false, false, true],
                &[false, false, false],
                &[0.0, 0.0, 0.0],
                1.0,
            ),
            vec![4.0, -4.0, 4.0]
        );
    }

    #[test]
    fn truncated_rollouts_bootstrap_from_the_next_value() {
        assert_eq!(
            compute_rewards_to_go(&[0.0], &[0], &[1], &[false], &[true], &[2.0], 0.99,),
            vec![-1.98]
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
    fn trainer_can_update_from_a_truncated_rollout() {
        let config = PpoConfig {
            timesteps_per_batch: 1,
            max_timesteps_per_episode: 1,
            updates_per_iteration: 1,
            ..PpoConfig::default()
        };
        let mut trainer = PpoTrainer::new(config).expect("trainer should initialize");
        let mut environment = AzulEnv::new(0, config.max_timesteps_per_episode);

        trainer.train(&mut environment, 1);
    }
}
