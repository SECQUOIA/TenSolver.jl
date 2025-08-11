function capture_stdout(f)
    stdout_orig = stdout
    (rd, wr) = redirect_stdout()
    f()
    close(wr)
    redirect_stdout(stdout_orig)
    return read(rd, String)
end

@testset "Logging utilities" begin
  out0 = capture_stdout() do
    minimize([1.0 0.0; 0.0 -1.0], 3.0; iterations = 1, verbosity = 0)
  end

  @test isempty(out0)

  out1 = capture_stdout() do
    minimize([1.0 0.0; 0.0 -1.0], 3.0; iterations = 1, verbosity = 1)
  end

  @test !isempty(out1)
end
