* rews_ext: Simply the reward given by the environment. No manipulation at hand
* 
* rews_int: The intrisic reward given by the architecture, unnormalized
* rews_int_norm: The intrisic reward given by the architecture. In contrast to the external rewards, these get      normalized by calculating the variance in them and taking the square-root of that
* rewintmean_norm: The mean over the normalized intrinsic rewards
* rewintstd_norm: The standrad deviation over the normalized intrinsic rewards
* rewintmax_norm: The maximum over the normalized intrinsic rewards
* rewintmean_unnorm: The mean over the unnormalized intrinsic rewrads
* rewintmax_unnorm: The maximum over the unnormalized intrinsic rewards
* 
* best_ret: The highest achieved external reward of all subthreads. Gets calculated episode-wise
* 
* ev_ext: Explained variance with regard to the networks value prediction(Read Paper) and the final gathered values
* ev_int: Explained variance with regard to the networks value prediction(Read Paper) and the final gathered values
* 
* adv_int: Intrinsic advantages
* adv_ext: Extrinsic advantages
* advmean: Mean over both types of advantages
* advstd: Standard deviation over both types of advantages

* ent: PROBABLY only the networks entropy

* ret_int: Sum of the networks value prediction(Read Paper) and the corresponding advantages
* retintmean: Mean over this very sum
* retintstd: Standard deviation over this very sum
* 
* ret_ext: Sum of the networks value prediction(Read Paper) and the corresponding advantages
* retextmean: Mean over this very sum
* retextstd: Standard deviation over this very sum

*  
* vpred_ext: The value networks prediction of the extrinsic rewards
* vpredextmean: The mean over this prediction
* vpredextstd: Standard deviation over this prediction
* 
* vpred_int: The value networks prediction of the intrinsic rewards
* vpredintmean: The mean over this prediction
* vrpedintstd: Standard deviation over this prediction
* 
* 
* 
* acs: A list of chosen actions for a random agent (Helpful to gather random statistics)
* visited_rooms: List of visited rooms, important for the Environment "Montezumas Revenge"
* n_rooms: Number of rooms visited, important for the Environment "Montezumas Revenge"
* reset_counter: Integer, possibly to count the number of environment resets. But is never actually manipulated
* Number of Episodes: Gatherer to count the number of run episodes
* Episode Reward: Episode reward(extrinsic) taken from the "info"-variable returned by an environment-step
* Episode Length: Episode length taken from the "info"-Argument returned by an environment-step