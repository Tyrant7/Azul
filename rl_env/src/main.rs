mod logging;

fn main() -> Result<(), tch::TchError> {
    let config = rl_env::PpoConfig {
        timesteps_per_batch: 1_000,
        updates_per_iteration: 5,
        gamma: 0.995,
        ..Default::default()
    };

    let mut trainer = rl_env::PpoTrainer::new(config)?;
    trainer.add_historical_opponent()?;
    let mut environment = rl_env::AzulEnv::new(0, None);
    let mut logger = logging::TrainingLogger::new("camf_spl_LR_stab_league");

    logger.log_device();
    trainer.train_with_callback(&mut environment, 1_000_000, |metrics| logger.log(metrics));
    std::fs::create_dir_all("checkpoints").expect("checkpoint directory should be creatable");
    trainer.save_checkpoints("checkpoints/azul_actor.ot", "checkpoints/azul_critic.ot")?;
    Ok(())
}
