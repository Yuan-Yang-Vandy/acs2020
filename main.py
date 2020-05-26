import run_raven
import strategy

# M-confident strategy
run_raven.run_raven(strategy.confident, prediction_score_name = "mato_score")

# M-neutral strategy
run_raven.run_raven(strategy.neutral, prediction_score_name = "mato_score")

# M-prudent strategy
run_raven.run_raven(strategy.prudent, prediction_score_name = "mato_score")

# O-confident strategy
run_raven.run_raven(strategy.confident, prediction_score_name = "optn_score")

# O-neutral strategy
run_raven.run_raven(strategy.neutral,  prediction_score_name = "optn_score")

# O-prudent strategy
run_raven.run_raven(strategy.prudent,  prediction_score_name = "optn_score")


