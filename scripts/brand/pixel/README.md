# Pixel pictures of the headline card

`accuracy.png` (Nest's robot and a bullseye), `cost.png` (one coin against a tower of coins) and `time.png` (the robot
running with a stopwatch), 336 × 168, indexed, ≤ 32 colours. Drawn with Nest's pixel pipeline (the robot and the city
of nest.technology): an image model redraws each scene from a prompt and two references, then a deterministic step fits
it to an exact pixel grid and palette. `headline_card.py` embeds them under the three final figures.
