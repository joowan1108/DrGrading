"""Train IDRiD with random adjacent margins and a longer, more patient schedule."""
from train_idrid import main


if __name__ == "__main__":
    main(default_config="configs/hybrid_idrid_random_margin_long.yaml")
