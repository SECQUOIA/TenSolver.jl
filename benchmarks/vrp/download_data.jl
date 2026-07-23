using Downloads
using SHA

const SOURCE_COMMIT = "78c1e390309127a6be09b11b599f25e435d9f324"
const SOURCE_URL =
  "https://raw.githubusercontent.com/SECQUOIA/vrp-qinnovision/$SOURCE_COMMIT/TestSet"

const INSTANCES = [
  (
    name = "test_pb_10_o.rudy",
    sha256 = "a6e63f5984c2d48d654544fdab8c1a61c005dc63c9bac7bc94af55ef204f15e8",
  ),
  (
    name = "test_pb_27_o.rudy",
    sha256 = "5342310d3bfdb069870892b25b2f07f53cb99a8dd3b0eb9d9ae83e132a27abb1",
  ),
  (
    name = "test_pb_49_o.rudy",
    sha256 = "f468f6e7a7cb252ee997993e1af84c83a07bbec463a3a3a840bbc9eb6a66c4bd",
  ),
  (
    name = "test_pb_96_o.rudy",
    sha256 = "0a765835e5a6f35ee5312ead5dcc43036dc2808d6bc0f15865c082886df53630",
  ),
  (
    name = "test_pb_217_o.rudy",
    sha256 = "d66e0d4e8ccc388406816e6460ff2d779d4866aba5a69896b8eb3eda16eb0475",
  ),
  (
    name = "test_pb_262_o.rudy",
    sha256 = "b8a208516bd39ff3a73c67d09a4565a03c75e76b6ebbfb58e229e1284e326c37",
  ),
  (
    name = "test_pb_541_o.rudy",
    sha256 = "191494c46d2d54d579f3eb3b33fad872c088c9c91a8f0af4b708e99796e857a0",
  ),
  (
    name = "test_pb_794_o.rudy",
    sha256 = "a33f091732a3c3b9c4cf882dce3f5806056f3c5a91951815388c27d7dfb4d458",
  ),
]

checksum(path) = bytes2hex(open(sha256, path))

function verify_checksum(path, expected)
  actual = checksum(path)
  actual == expected ||
    error("Checksum mismatch for $(basename(path)): expected $expected, got $actual")
  return nothing
end

function download_instance(data_dir, instance)
  destination = joinpath(data_dir, instance.name)
  if isfile(destination)
    verify_checksum(destination, instance.sha256)
    println("verified $(instance.name)")
    return nothing
  end

  mktemp(data_dir) do temporary, io
    close(io)
    Downloads.download("$(SOURCE_URL)/$(instance.name)", temporary)
    verify_checksum(temporary, instance.sha256)
    mv(temporary, destination)
  end
  println("downloaded $(instance.name)")
  return nothing
end

function main(args = ARGS)
  length(args) <= 1 || error("usage: julia download_data.jl [destination]")
  data_dir = isempty(args) ? joinpath(@__DIR__, "data") : abspath(only(args))
  mkpath(data_dir)
  foreach(instance -> download_instance(data_dir, instance), INSTANCES)
  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
