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
}
