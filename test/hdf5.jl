using TenSolver, HDF5, ITensorMPS

@testset "HDF5 snapshot callback" begin
  n_vars   = 10
  iter     = 6
  cb_every = 2

  Q = randn(n_vars, n_vars)

  tmpdir         = mktempdir()
  snapshots_path = joinpath(tmpdir, "snapshots.h5")

  function cb(mps; iteration, kw...)
    h5open(snapshots_path, "cw") do f
      g = create_group(f, "iter_$iteration")
      write(g, "mps", mps)
    end
  end

  E, psi = TenSolver.minimize(Q; iterations=iter, on_iteration=cb, callback_every=cb_every)

  expected_iters = cb_every:cb_every:iter   # [2, 4, 6]

  @testset "Correct number of groups written" begin
    h5open(snapshots_path, "r") do f
      @test length(keys(f)) == length(expected_iters)
    end
  end

  @testset "Iter $i: group exists, MPS readable, samples valid" for i in expected_iters
    mps = h5open(snapshots_path, "r") do f
      @test haskey(f, "iter_$i")
      read(f["iter_$i"], "mps", MPS)
    end

    @test mps isa MPS
    @test length(mps) == n_vars

    xs = [ITensorMPS.sample!(mps) .- 1 for _ in 1:20]
    @test all(x -> length(x) == n_vars,           xs)
    @test all(x -> all(b -> b == 0 || b == 1, x), xs)
  end
end
