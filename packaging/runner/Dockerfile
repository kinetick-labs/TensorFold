ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG RUNNER_BUILD_ID=runner-frontend
LABEL org.tensorfold.runner.frontend.build=${RUNNER_BUILD_ID}
RUN nice -n 19 sh -ec 'apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-18 lld-18; ln -sf clang-18 /usr/bin/clang; ln -sf ld.lld-18 /usr/bin/ld.lld; apt-get clean; find /var/lib/apt/lists -type f -delete'
COPY tf-runner-frontend-check /opt/tensorfold/bin/tf-runner-frontend-check
COPY runner-entrypoint /opt/tensorfold/bin/runner-entrypoint
RUN nice -n 19 /opt/tensorfold/bin/tf-runner-frontend-check --require
ARG FRONTEND_TREE
LABEL org.tensorfold.runner.frontend.tree=${FRONTEND_TREE}
ENTRYPOINT ["/opt/tensorfold/bin/runner-entrypoint"]
