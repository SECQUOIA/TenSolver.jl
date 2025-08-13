import Printf: @printf

function iterlog_horizontal_rule()
  @printf("+-----------+-------------+----------+--------------+------------+\n")
end

function iterlog_header(verbosity)
  if verbosity > 0
    iterlog_horizontal_rule()
    @printf("| Iteration |  Objective  | Max Bond |   Variance   |  Time (s)  |\n")
    iterlog_horizontal_rule()
  end
end

function iterlog_iteration(verbosity, iter, obj, bond_dim, variance, elapsed_time)
  if verbosity > 0
    @printf(
      "| %9d | %# 11.4g | % 8d | %# 12.4e | %#10.2g |\n",
      iter,
      obj,
      bond_dim,
      variance,
      elapsed_time,
    )
  end
end

function iterlog_iteration(verbosity, iter, obj, bond_dim, variance::Nothing, elapsed_time)
  if verbosity > 0
    @printf(
      "| %9d | %# 11.4g | % 8d | %12s | %#10.2g |\n",
      iter,
      obj,
      bond_dim,
      "     -     ",
      elapsed_time,
    )
  end
end

function iterlog_footer(verbosity, obj, elapsed_time)
  if verbosity > 0
    iterlog_horizontal_rule()
    @printf("\n")
    @printf("Objective value : %-# .4g\n", obj)
    @printf("Total time      : %-# .2g s\n", elapsed_time)
  end
end

