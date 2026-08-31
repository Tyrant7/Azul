use std::ops::Mul;

use tch::{
    Kind,
    Reduction::Mean,
    nn::{Module, VarStore},
};

use crate::{
    AzulEnv, ReplayBuffer, Transition, encode_gamestate, get_device,
    net::{ResNetwork, initialize_actor, initialize_critic},
};

pub fn ppo() {
    let actor_vs = VarStore::new(get_device());
    let critic_vs = VarStore::new(get_device());
    let actor = initialize_actor(&actor_vs.root());
    let critic = initialize_critic(&critic_vs.root());

    let mut env = AzulEnv::new(rand::random(), 1000);

    let max_steps = 10000;
    let mut steps = 0;
    while steps < max_steps {
        rollout(&mut env, &actor);
        steps += 1;
    }
}

pub fn rollout(env: &mut AzulEnv, actor: &ResNetwork) {
    let mut t = 0;
    let mut buffer = ReplayBuffer::new(10000);

    let timesteps_per_batch = 1000;
    while t < timesteps_per_batch {
        env.reset(env.max_steps);

        for episode in 0..env.max_steps {
            t += 1;

            let action_mask = env.action_mask().unsqueeze(0);
            let state = encode_gamestate(&env.gamestate);
            let mean = actor.forward(&state.unsqueeze(0));
            let probs = mean.softmax(0, Kind::Float).mul(action_mask);
            let action = probs.multinomial(1, false).int64_value(&[0, 0]) as usize;
            let log_probs = probs.log();

            let result = env.step(action).expect("illegal move played");

            buffer.push(Transition::new(
                &state,
                action,
                &log_probs,
                result.reward,
                &encode_gamestate(&env.gamestate),
                result.terminated,
                result.truncated,
            ));

            if result.terminated {
                break;
            }
        }
    }
}
