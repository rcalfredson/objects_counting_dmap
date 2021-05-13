import pstats

p = pstats.Stats('fcrn_a_performance_stats')
p.sort_stats('cumulative').print_stats(10)
