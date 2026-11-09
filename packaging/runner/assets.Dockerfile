# The runner's assets over an image build.sh layered (the circuit frontend: clang-18 / lld-18 and
# tf-runner-frontend-check), supplied by build-assets.sh: the runner circuit and the graph bank assets.sha256 pins.
# It adds no engine binary, and the build fails unless the image's circuit route computes the engine's probe cells.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY runner/calc.tfpc /opt/tensorfold/share/runner/calc.tfpc
COPY runner/graph-v7-powmod/bank/ /opt/tensorfold/share/runner/graph-v7-powmod/bank/
COPY runner/graph-v7-powmod/bank/Qwenputer-MIT.txt /opt/tensorfold/licenses/runtime-deps/Qwenputer-MIT.txt
RUN nice -n 19 /opt/tensorfold/bin/tf-runner-frontend-check --require --circuit /opt/tensorfold/share/runner/calc.tfpc
LABEL org.tensorfold.runner.circuit="calc.tfpc" org.tensorfold.runner.graph="graph-v7-powmod"
