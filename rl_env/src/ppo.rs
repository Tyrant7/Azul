use tch::nn::VarStore;

use crate::{
    get_device,
    net::{initialize_actor, initialize_critic},
};

pub fn ppo() {
    let actor_vs = VarStore::new(get_device());
    let critic_vs = VarStore::new(get_device());
    let actor = initialize_actor(&actor_vs.root());
    let critic = initialize_critic(&critic_vs.root());

    let max_steps = 10000;
    let mut steps = 0;
    while steps < max_steps {
        steps += 1;
    }
}
