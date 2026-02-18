cd bandit

# ours
# for i in {1..3}; do
SEED=$RANDOM
    python main.py --method mepoly --task lemniscate --ent_coef=0.2 --poly_order=14 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
    python main.py --method mepoly --task twomoons --ent_coef=0.2 --poly_order=14 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# done


# ppo-gaussian
# for i in {1..3}; do
SEED=$RANDOM
    python main.py --method ppo --task lemniscate --ent_coef=0.2  --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
    python main.py --method ppo --task twomoons --ent_coef=0.2 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# done

# ppo-gmm
# for i in {1..3}; do
SEED=$RANDOM
    python main.py --method gmm --task lemniscate --ent_coef=0.1  --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
    python main.py --method gmm --task twomoons --ent_coef=0.1 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# done

# fpo
# for i in {1..3}; do
SEED=$RANDOM
    python main.py --method fpo --task lemniscate --n_updates_per_iteration=10 --num_fpo_samples=2000 --max_iterations=100 --seed=${SEED}
    python main.py --method fpo --task twomoons --n_updates_per_iteration=10 --num_fpo_samples=2000 --max_iterations=100 --seed=${SEED}
# done

# ablation 1: order
# SEED=$RANDOM
# python main.py --method mepoly --task lemniscate --ent_coef=0.2 --poly_order=2 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# python main.py --method mepoly --task lemniscate --ent_coef=0.2 --poly_order=6 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# python main.py --method mepoly --task lemniscate --ent_coef=0.2 --poly_order=10 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# python main.py --method mepoly --task lemniscate --ent_coef=0.2 --poly_order=14 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# python main.py --method mepoly --task twomoons --ent_coef=0.2 --poly_order=2 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# python main.py --method mepoly --task twomoons --ent_coef=0.2 --poly_order=6 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# python main.py --method mepoly --task twomoons --ent_coef=0.2 --poly_order=10 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}
# python main.py --method mepoly --task twomoons --ent_coef=0.2 --poly_order=14 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=100 --seed=${SEED}


# ablation 2: legendre
# comment out legendre basis in polynomials.py before running
# SEED=$RANDOM
# python main.py --method mepoly --task twomoons --ent_coef=0.6 --poly_order=12 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=600 --seed=${SEED}
# python main.py --method mepoly --task lemniscate --ent_coef=0.6 --poly_order=12 --poly_grid_size=256 --lambda_clip=10000 --n_updates_per_iteration=10 --max_iterations=600 --seed=${SEED}