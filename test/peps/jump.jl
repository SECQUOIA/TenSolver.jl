import JuMP, QUBODrivers, QUBOTools

@testset "JuMP SpinGlassPEPS CPU integration" begin
  # Two independent pairs: (x1, x2) has minimum -1 at (1, 0),
  # and (x3, x4) has minimum -1 at (1, 1). Both have maximum 0 at (0, 0).
  # The diagonal terms and offset exercise Boolean-to-spin conversion.
  Q = [
    -1.0 0.5 0.0 0.0
    0.0 -0.5 0.0 0.0
    0.0 0.0 -0.25 0.25
    0.0 0.0 0.0 -0.75
  ]
  l = [0.0, 0.25, -0.25, 0.0]
  c = 0.125
  objective(state) = dot(state, Q, state) + dot(l, state) + c
  @test objective([1, 0, 1, 1]) == -1.875
  @test objective([0, 0, 0, 0]) == 0.125
  @test first(brute_force(objective, 4)) == -1.875
  @test -first(brute_force(state -> -objective(state), 4)) == 0.125

  for layout in (:square, :king), sense in (JuMP.MOI.MIN_SENSE, JuMP.MOI.MAX_SENSE)
    @testset "$layout $sense" begin
      minimizing = sense == JuMP.MOI.MIN_SENSE
      expected_energy = minimizing ? -1.875 : 0.125
      optimum = minimizing ? [1, 0, 1, 1] : [0, 0, 0, 0]
      beta = 2.0
      final_reads = 37
      model = JuMP.Model(TenSolver.Optimizer)
      JuMP.set_attribute(model, "verbosity", 0)
      JuMP.set_attribute(model, "backend", " PEPS ")
      JuMP.set_attribute(model, "peps_layout", layout)
      JuMP.set_attribute(model, "peps_topology", (2, 2))
      JuMP.set_attribute(model, "peps_beta", beta)
      JuMP.set_attribute(model, "peps_bond_dim", 4)
      JuMP.set_attribute(model, "peps_max_states", 16)
      JuMP.set_attribute(model, "peps_cutoff_prob", 0.0)
      JuMP.set_attribute(model, "peps_strategy", :svd)
      JuMP.set_attribute(model, "peps_transformations", :identity)
      JuMP.set_attribute(model, "num_reads", 11)
      JuMP.set_attribute(model, QUBODrivers.FinalNumberOfReads(), final_reads)
      JuMP.@variable(model, x[1:4], Bin)
      JuMP.set_objective_sense(model, sense)
      JuMP.set_objective_function(model, dot(x, Q, x) + dot(l, x) + c)

      JuMP.optimize!(model)

      @test JuMP.objective_value(model) ≈ expected_energy atol = 1e-6
      @test round.(Int, JuMP.value.(x)) == optimum
      solution = QUBOTools.solution(JuMP.unsafe_backend(model))
      @test QUBOTools.reads(solution) == final_reads
      @test QUBOTools.state(solution, 1) == optimum
      @test isempty(QUBODrivers.validate_metadata(solution))
      metadata = QUBOTools.metadata(solution)
      peps = metadata["tensolver"]["peps"]
      @test metadata["algorithm"]["name"] == "SpinGlassPEPS"
      @test metadata["backend"]["name"] == "TenSolver"
      @test metadata["backend"]["version"] == TenSolver.__VERSION__
      @test metadata["reads"]["number_of_reads"] == 11
      @test metadata["reads"]["final_number_of_reads"] == final_reads
      @test metadata["optimizer"]["evaluations"] == peps["candidate_states"]
      @test 1 <= peps["candidate_states"] <= 16
      @test peps["backend"] == "SpinGlassPEPS"
      @test peps["topology"] == string(layout)
      @test peps["topology_size"] == (2, 2, 1)
      @test peps["selected_transformation"] == string(SpinGlassPEPS.rotation(0))
      @test haskey(peps, "largest_discarded_probability")
      @test haskey(peps, "raw")
      @test all(<=(0), peps["spin_glass_probabilities"])
      @test peps["effective_time"] >= 0
      @test peps["parameters"]["beta"] == beta
      @test peps["parameters"]["bond_dim"] == 4
      @test peps["parameters"]["max_states"] == 16
      @test peps["parameters"]["cutoff_prob"] == 0.0
      @test peps["parameters"]["strategy"] == "svd"
      @test peps["parameters"]["transformations"] == :identity

      # Preserve all retained Boolean states and their conditional probabilities,
      # including states whose read allocation rounds to zero.
      states = peps["states"]
      probabilities = peps["probabilities"]
      @test length(states) == length(probabilities) == peps["candidate_states"]
      @test allunique(states)
      @test all(state -> length(state) == 4 && all(in((0, 1)), state), states)
      @test optimum in states
      @test all(isfinite, probabilities)
      @test all(>=(0), probabilities)
      @test sum(probabilities) ≈ 1.0
      # Upstream branch merging can retain fewer than max_states. Check the
      # Boltzmann distribution conditioned on the states actually retained.
      direction = minimizing ? 1 : -1
      weights =
        [exp(-beta * direction * (objective(state) - expected_energy)) for state in states]
      @test probabilities ≈ weights ./ sum(weights) atol = 1e-6
      @test length(solution) <= length(states)
      for (state, probability) in zip(states, probabilities)
        index = findfirst(i -> QUBOTools.state(solution, i) == state, 1:length(solution))
        reads = isnothing(index) ? 0 : QUBOTools.reads(solution, index)
        @test abs(reads - final_reads * probability) <= 1 + 1e-6
      end
      for i in 1:length(solution)
        state = QUBOTools.state(solution, i)
        @test state in states
        @test objective(state) ≈ QUBOTools.value(solution, i) atol = 1e-6
        @test QUBOTools.reads(solution, i) > 0
      end
    end
  end
end
