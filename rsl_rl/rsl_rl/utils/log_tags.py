"""TensorBoard presentation groups; metric values and aggregation stay unchanged."""


def episode_scalar_tag(key):
    if key.startswith(("gait_height/", "gait_retraction/", "retraction_other/")):
        return "temp/" + key
    if key.startswith("rew_"):
        return "Episode/" + key
    return "others/" + key


def diagnostic_scalar_tag(key):
    return "others/" + key
