

for i in {1..3}; do
    SEED=$RANDOM
    python smoothworld/main.py --method mepoly --poly_order=5 --ent_coef=0.10 --lambda_clip=5 --max_iterations=500 --poly_grid_size=64 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=two_walls
    SEED=$RANDOM
    python smoothworld/main.py --method mepoly --poly_order=5 --ent_coef=0.10 --lambda_clip=5 --max_iterations=2000 --poly_grid_size=64 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=four_walls
    SEED=$RANDOM
    python smoothworld/main.py --method mepoly --poly_order=5 --ent_coef=0.05 --lambda_clip=5 --max_iterations=6000 --poly_grid_size=64 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=cshape
    SEED=$RANDOM
    python smoothworld/main.py --method mepoly --poly_order=5 --ent_coef=0.05 --lambda_clip=5 --max_iterations=4000 --poly_grid_size=64 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=tree_in_the_middle
    SEED=$RANDOM
    python smoothworld/main.py --method mepoly --poly_order=5 --ent_coef=0.10 --lambda_clip=5 --max_iterations=4000 --poly_grid_size=64 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=two_slits
done


for i in {1..3}; do
    SEED=$RANDOM
    python smoothworld/main.py --method fpo --num_fpo_samples=50  --max_iterations=500 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=two_walls
    SEED=$RANDOM
    python smoothworld/main.py --method fpo --num_fpo_samples=50 --max_iterations=2000 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=four_walls
    SEED=$RANDOM
    python smoothworld/main.py --method fpo --num_fpo_samples=50 --max_iterations=4000 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=two_slits
    SEED=$RANDOM
    python smoothworld/main.py --method fpo --num_fpo_samples=50 --max_iterations=6000 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=cshape
    SEED=$RANDOM
    python smoothworld/main.py --method fpo --num_fpo_samples=50 --max_iterations=4000 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=tree_in_the_middle
done


for i in {1..3}; do
    SEED=$RANDOM
    python smoothworld/main.py --method gmm --ent_coef=0.0010 --max_iterations=500 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=two_walls
    SEED=$RANDOM
    python smoothworld/main.py --method gmm --ent_coef=0.0010 --max_iterations=2000 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=four_walls
    SEED=$RANDOM
    python smoothworld/main.py --method gmm --ent_coef=0.0010 --max_iterations=4000 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=two_slits
    SEED=$RANDOM
    python smoothworld/main.py --method gmm --ent_coef=0.0005 --max_iterations=6000 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=cshape
    SEED=$RANDOM
    python smoothworld/main.py --method gmm --ent_coef=0.0005 --max_iterations=4000 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=tree_in_the_middle
done


for i in {1..3}; do
    SEED=$RANDOM
    python smoothworld/main.py --method ppo --ent_coef=0.0010 --max_iterations=500 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=two_walls
    SEED=$RANDOM
    python smoothworld/main.py --method ppo --ent_coef=0.0010 --max_iterations=2000 --n_updates_per_iteration=10 --seed=${SEED} --grid_mode=four_walls
    SEED=$RANDOM
    python smoothworld/main.py --method ppo --ent_coef=0.0010 --max_iterations=4000 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=two_slits
    SEED=$RANDOM
    python smoothworld/main.py --method ppo --ent_coef=0.0005 --max_iterations=6000 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=cshape
    SEED=$RANDOM
    python smoothworld/main.py --method ppo --ent_coef=0.0005 --max_iterations=4000 --n_updates_per_iteration=1 --seed=${SEED} --grid_mode=tree_in_the_middle
done