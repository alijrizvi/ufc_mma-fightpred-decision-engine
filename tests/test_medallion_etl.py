from src.engine.etl import OctagonDataEngine


EXPECTED_FIGHTERS = [
    "Islam Makhachev",
    "Carlos Prates",
    "Justin Gaethje",
    "Dustin Poirier",
    "Charles Oliveira",
    "Ilia Topuria",
]


def test_medallion_star_schema_all_six_fighters():
    """Verifies Bronze -> Silver -> Gold Star Schema populates all 6 target fighters and fact joins."""
    engine = OctagonDataEngine()
    engine.seed_historical_lightweight_dataset()

    summary = engine.get_star_schema_summary()
    counts = summary["table_counts"]
    assert counts["bronze.raw_fighters"] == 6
    assert counts["silver.stg_fighters"] == 6
    assert counts["gold.dim_fighter"] == 6
    assert counts["gold.dim_bout"] == 3
    assert counts["gold.dim_round_interval"] == 150
    assert counts["gold.fact_interval_telemetry"] == 8

    for name in EXPECTED_FIGHTERS:
        profile = engine.load_fighter_profile(name)
        assert profile.name == name
        assert profile.punch_power > 0
        metrics = engine.get_fighter_pillar_metrics(name)
        assert 0.0 <= metrics["strike_hit_pct"] <= 1.0


def test_silver_layer_cleans_and_deduplicates_dirty_bronze_data():
    """Verifies Silver deduplicates fighters and clamps out-of-range percentages."""
    engine = OctagonDataEngine()
    dirty_fighters = [
        ("  Ilia Topuria ", "Featherweight", "Orthodox", 31.0, 23.0, 17.0, 1.45, -0.2, 0.38, 0.92, 0.92, 0.56, 15.0, 0.28, 0.028),
        ("Ilia Topuria", "Featherweight", "Orthodox", 31.0, 23.0, 17.0, 0.56, 0.46, 0.38, 0.92, 0.92, 0.56, 15.0, 0.28, 0.028),
    ]
    intervals = [
        ("BOUT-1", 1, "Ilia Topuria", "Ilia Topuria", 1, 10, 15, 20.0, 1, 1, 2, 5, 3, 6),
    ]
    engine.ingest_bronze_records(dirty_fighters, intervals, source_system="dirty_test")
    engine.transform_bronze_to_silver()
    engine.build_gold_star_schema()

    prof = engine.load_fighter_profile("Ilia Topuria")
    assert 0.0 <= prof.strike_hit_pct <= 1.0
    assert 0.0 <= prof.arm_block_pct <= 1.0
    assert engine.get_star_schema_summary()["table_counts"]["gold.dim_fighter"] == 1
