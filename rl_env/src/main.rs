mod logging;

const FINAL_EVALUATION_GAMES: usize = 500;

fn main() -> Result<(), tch::TchError> {
    let config = rl_env::PpoConfig {
        timesteps_per_batch: 1_000,
        updates_per_iteration: 4,
        gamma: 0.99,
        evaluation_games: 16,
        evaluation_interval: 10,
        ..Default::default()
    };

    let mut trainer = rl_env::PpoTrainer::new(config)?;
    trainer.set_reference_actor("checkpoints/reference_actor.ot")?;
    let mut environment = rl_env::AzulEnv::new(0, None);
    let mut logger = logging::TrainingLogger::new("full_league_more_features");

    logger.log_device();
    trainer.train_with_callback(&mut environment, 1_000_000, |metrics| logger.log(metrics));
    let final_evaluation = trainer.evaluate_greedy_against_reference(
        &mut environment,
        FINAL_EVALUATION_GAMES,
        config.evaluation_seed.wrapping_add(1_000_000),
    );
    println!(
        "final_evaluation games={} wins={} losses={} win_rate={:.3}",
        final_evaluation.games,
        final_evaluation.wins,
        final_evaluation.losses,
        final_evaluation.win_rate,
    );
    std::fs::create_dir_all("checkpoints").expect("checkpoint directory should be creatable");
    trainer.save_checkpoints("checkpoints/azul_actor.ot", "checkpoints/azul_critic.ot")?;
    Ok(())
}
