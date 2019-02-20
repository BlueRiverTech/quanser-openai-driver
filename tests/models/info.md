# Reward function 1

### Up env:
This used a reward of `1 - abs(alpha) - 0.1*abs(theta)` to solve the `QubeBeginUprightEnv` problem to learn hold.


### Switched from up to down:

- Used the pre-trained model from the up env (after 1 million steps), then swtiched to the `QubeBeginDownEnv` to learn flip up + old.
- This worked well but kept hitting the sides, after 1.6M it went too far towards the edges. A greater penalty on theta was needed.
- Best performing models are 1.5M and 1.6M steps of learning to flip up.


### Added a penalty on theta

- Increased the penalty on theta (and decreased the penalty on alpha to ensure the minimum reward does not go below 0)
- After only 100K steps on this new environment, the Qube learned a good stable controller for combined flip up + hold